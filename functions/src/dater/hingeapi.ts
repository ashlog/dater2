import {FeedsResponse, Message, Profile} from './types';

export interface HingeAPI {
  getProfiles(ids: string[]): Promise<Profile[]>;

  sendLike(
    subjectId: string,
    ratingToken: string,
    sessionId: string,
    options: {
      photoData?: {url: string; cdnId: string};
      content?: {prompt: {question: string; answer: string}};
      comment?: string;
    }
  ): Promise<void>;

  getRecommendations(
    longitude: number,
    latitude: number
  ): Promise<FeedsResponse>;

  sendMessage(input: Message[]): Promise<void>;
}
