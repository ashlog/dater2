import axios from 'axios';
import {FeedsResponse, Message, Profile} from './types';
import {v4 as uuidv4} from 'uuid';
import {HingeAPI} from './hingeapi';

export class HingeAPIImpl implements HingeAPI {
  private host = 'prod-api.hingeaws.net';
  private baseURL = `https://${this.host}`;

  constructor(
    private authToken: string,
    private deviceHeaders: {[key: string]: string},
    private playerId?: string
  ) {}

  private getHeaders() {
    return {
      Host: this.host,
      Authorization: `Bearer ${this.authToken}`,
      Accept: '*/*',
      ...this.deviceHeaders,
    };
  }

  async getProfiles(ids: string[]): Promise<Profile[]> {
    const url = `${this.baseURL}/user/v2/public`;
    const params = {
      ids: ids.join(','),
    };

    return (
      await axios.get<Profile[]>(url, {headers: this.getHeaders(), params})
    ).data;
  }

  async textReview(receiverId: string, text: string): Promise<{hcmRunId: string; isHarmful: boolean}> {
    const url = `${this.baseURL}/flag/textreview`;
    const response = await axios.post<{hcmRunId: string; isHarmful: boolean}>(url, {receiverId, text}, {
      headers: this.getHeaders(),
    });
    return response.data;
  }

  async sendLike(
    subjectId: string,
    ratingToken: string,
    sessionId: string,
    options: {
      photoData?: {url: string; cdnId: string};
      content?: {prompt: {contentId?: string; question: string; answer: string}};
      comment?: string;
      hcmRunId?: string;
    }
  ): Promise<void> {
    const url = `${this.baseURL}/rate/v2/initiate`;

    const contentValue = options.content
      ? options.content
      : {
          photo: {
            boundingBox: {
              bottomRight: {
                y: 1,
                x: 1,
              },
              topLeft: {
                x: 0,
                y: 0,
              },
            },
            url: options.photoData!.url,
            cdnId: options.photoData!.cdnId,
          },
        };

    const data: Record<string, unknown> = {
      subjectId,
      sessionId: sessionId,
      content: options.comment ? {...contentValue, comment: options.comment} : contentValue,
      rating: 'note',
      ratingId: uuidv4().toUpperCase(),
      ratingToken: ratingToken,
      hasPairing: false,
      origin: 'compatibles',
      initiatedWith: 'standard',
      created: new Date().toISOString(),
    };

    if (options.hcmRunId) {
      data.hcmRunId = options.hcmRunId;
    }

    await axios.post(url, data, {
      headers: this.getHeaders(),
    });
  }

  async getRecommendations(
    longitude: number,
    latitude: number
  ): Promise<FeedsResponse> {
    const data = {
      activeToday: false,
      playerId: this.playerId,
      newHere: false,
    };

    const response = await axios.post<FeedsResponse>(
      `${this.baseURL}/rec/v2`,
      data,
      {
        headers: this.getHeaders(),
      }
    );
    return response.data;
  }

  async sendMessage(input: Message[]): Promise<void> {
    const headers = this.getHeaders();
    const url = `${this.baseURL}/message/send`;

    await axios.post(url, input, {
      headers,
    });
  }
}
