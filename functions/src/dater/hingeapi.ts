import {FeedsResponse, Message, Profile} from './types';

export interface HingeAPI {
  getProfiles(ids: string[]): Promise<Profile[]>;

  textReview(receiverId: string, text: string): Promise<{hcmRunId: string; isHarmful: boolean}>;

  sendLike(
    subjectId: string,
    ratingToken: string,
    sessionId: string,
    options: {
      photoData?: {url: string; cdnId: string};
      content?: {prompt: {question: string; answer: string}};
      comment?: string;
      hcmRunId?: string;
    }
  ): Promise<void>;

  getRecommendations(
    longitude: number,
    latitude: number
  ): Promise<FeedsResponse>;

  sendMessage(input: Message[]): Promise<void>;
}
