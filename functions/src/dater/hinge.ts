import axios from 'axios';
import {FeedsResponse, Message, Profile} from './types';
import {v4 as uuidv4} from 'uuid';
import {HingeAPI} from './hingeapi';

export class HingeAPIImpl implements HingeAPI {
  private host = 'prod-api.hingeaws.net';
  private baseURL = `https://${this.host}`;

  constructor(
    private authToken: string,
    private deviceHeaders: {[key: string]: string}
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

  async sendLike(
    subjectId: string,
    ratingToken: string,
    options: {
      photoData?: {url: string; cdnId: string};
      content?: {prompt: {question: string; answer: string}};
      comment?: string;
    }
  ): Promise<void> {
    const url = `${this.baseURL}/rate/v1/initiate`;

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

    const data = {
      subjectId,
      content: {...contentValue, comment: options.comment},
      rating: 'like',
      ratingId: uuidv4().toUpperCase(),
      ratingToken: ratingToken,
      hasPairing: false,
      origin: 'compatibles',
      initiatedWith: 'standard',
      created: new Date().toISOString(),
    };

    await axios.post(url, data, {
      headers: this.getHeaders(),
    });
  }

  async getRecommendations(
    longitude: number,
    latitude: number
  ): Promise<FeedsResponse> {
    const data = {
      excludedFeedIds: [],
      longitude: longitude,
      latitude: latitude,
      genderPrefId: 1,
      genderId: 0,
    };

    const response = await axios.post<FeedsResponse>(
      `${this.baseURL}/rec`,
      data,
      {
        headers: this.getHeaders(),
      }
    );
    return response.data;
  }

  async sendMessage(input: Message[]): Promise<void> {
    const headers = this.getHeaders();
    const url = `${this.baseURL}/message/chat`;

    await axios.post(url, input, {
      headers,
    });
  }
}
