import {FeedsResponse, Message, Profile} from './types';
import {HingeAPI} from './hingeapi';
import profiles from '../tests/profiles.json';
import recommendations from '../tests/recommendations.json';

export class FakeHingeAPI implements HingeAPI {
  constructor(
    private authToken: string,
    private deviceHeaders: {[key: string]: string}
  ) {}

  async getProfiles(ids: string[]): Promise<Profile[]> {
    return profiles as never as Profile[];
  }

  async sendLike(
    subjectId: string,
    ratingToken: string,
    sessionId: string,
    options: {
      photoData?: {url: string; cdnId: string};
      content?: {prompt: {question: string; answer: string}};
      comment?: string;
    }
  ): Promise<void> {}

  async getRecommendations(
    longitude: number,
    latitude: number
  ): Promise<FeedsResponse> {
    return recommendations as never as FeedsResponse;
  }

  async sendMessage(input: Message[]): Promise<void> {}
}
