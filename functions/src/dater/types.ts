export interface Profile {
  identityId: string;
  profile: {
    age: number;
    answers: [
      {
        position: number;
        questionId: string;
        type: string;
        response: string;
      }
    ];
    firstName: string;
    genderId: number;
    height: number;
    instafeedVisible: boolean;
    justJoinedDisplayUntil: string;
    languagesSpoken?: number[];
    languagesSpokenText?: string;
    lastName?: string;
    location: {
      name: string;
    };
    religionsText?: string;
    userId: string;
    lastActiveStatus: string;
    lastActiveStatusId: number;
    zodiac?: number;
    selfieVerified: boolean;
    children?: number;
    covidVax?: number;
    datingIntention?: number;
    datingIntentionText?: string;
    drinking: number;
    drugs: number;
    familyPlans?: number;
    marijuana: number;
    photos: [
      {
        cdnId: string;
        source: string;
        sourceId: string;
        boundingBox: {
          bottomRight: {
            x: number;
            y: number;
          };
          topLeft: {
            x: number;
            y: number;
          };
        };
        caption: string;
        location: string;
        promptId: string;
        videos: [
          {
            height?: number;
            width?: number;
            url: string;
            quality?: string;
          }
        ];
        url: string;
        width: number;
        height: number;
      }
    ];
    religions?: number[];
    smoking: number;
    jobTitle?: string;
    jobTitleText?: string;
    politicsText?: string;
    genderIdentity?: string;
    genderIdentityId?: number;
    hometown?: string;
  };
}

export interface PersonPreferences {
  excludedFeedIds: number[];
  longitude: number;
  latitude: number;
  genderPrefId: number;
  genderId: number;
}

export interface FeedsResponse {
  feeds: Feed[];
}

export interface Feed {
  id: number;
  origin: string;
  subjects: Subject[];
  preview: Preview;
}

export interface Preview {
  subjects: Subject[];
}

export interface Subject {
  subjectId: string;
  pairing: any;
  ratingToken: string;
}

export interface Message {
  isMatchMessage: boolean;
  recipient: string;
  text: string;
  created: string;
  messageId: string;
}

export interface GetMessagesParams {
  prev_limit: number;
  message_ts: number;
  include_reactions: boolean;
  include: boolean;
  with_sorted_meta_array: boolean;
  next: string;
  next_limit: number;
  reverse: boolean;
  show_subchannel_messages_only: boolean;
  include_reply_type: string;
  include_parent_message_info: boolean;
  include_thread_info: boolean;
  is_sdk: boolean;
  include_poll_details: boolean;
}

export interface SendBirdMessage {
  type: string;
  message_id: number;
  message: string;
  data: string;
  custom_type: string;
  file: object;
  created_at: number;
  user: User;
  channel_url: string;
  updated_at: number;
  message_survival_seconds: number;
  mentioned_users: any[];
  mention_type: string;
  silent: boolean;
  message_retention_hour: number;
  channel_type: string;
  translations: object;
  is_removed: boolean;
  req_id: string;
  is_op_msg: boolean;
  reactions: any[];
  message_events: MessageEvents;
}

export interface User {
  user_id: string;
  profile_url: string;
  require_auth_for_profile_image: boolean;
  nickname: string;
  metadata: object;
  role: string;
  is_active: boolean;
  is_blocked_by_me: boolean;
}

export interface MessageEvents {
  send_push_notification: string;
  update_unread_count: boolean;
  update_mention_count: boolean;
  update_last_message: boolean;
}

export interface HingeMessagesResponse {
  messages: SendBirdMessage[];
}

export interface SendBirdMessagesResponse {
  message_id: string;
  created_at: number;
}
