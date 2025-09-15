export interface Channel2 {
  channel_url: string;
  name: string;
  cover_url: string;
  data: string;
  created_at: number;
  custom_type: string;
  max_length_message: number;
  member_count: number;
}

export interface CreatedBy {
  user_id: string;
  nickname: string;
  profile_url: string;
  require_auth_for_profile_image: boolean;
}

export interface DisappearingMessage {
  is_triggered_by_message_read: boolean;
  message_survival_seconds: number;
}

export interface ReadReceipt {
  [key: number]: number;
}

export interface SmsFallback {
  wait_seconds: number;
  exclude_user_ids: any[];
}

export interface Inviter {
  user_id: string;
  nickname: string;
  profile_url: string;
  require_auth_for_profile_image: boolean;
  metadata: any;
}

export interface User {
  user_id: string;
  profile_url: string;
  require_auth_for_profile_image: boolean;
  nickname: string;
  metadata: any;
  is_blocked_by_me: boolean;
  role: string;
  is_active: boolean;
}

export interface MessageEvents {
  send_push_notification: string;
  update_unread_count: boolean;
  update_mention_count: boolean;
  update_last_message: boolean;
}

export interface LastMessage {
  type: string;
  message_id: any;
  message: string;
  data: string;
  custom_type: string;
  file: File;
  created_at: any;
  user: User;
  channel_url: string;
  updated_at: number;
  message_survival_seconds: number;
  mentioned_users: any[];
  mention_type: string;
  silent: boolean;
  message_retention_hour: number;
  channel_type: string;
  translations: any;
  is_removed: boolean;
  req_id: string;
  is_op_msg: boolean;
  message_events: MessageEvents;
}

export interface Member {
  user_id: string;
  nickname: string;
  profile_url: string;
  require_auth_for_profile_image: boolean;
  metadata: any;
  is_online: boolean;
  last_seen_at: number;
  state: string;
  is_active: boolean;
  is_blocked_by_me: boolean;
  friend_name?: any;
  friend_discovery_key?: any;
  role: string;
  is_muted: boolean;
  muted_end_at: number;
  muted_description: string;
  is_blocking_me: boolean;
}

export interface Channel {
  channel_url: string;
  name: string;
  cover_url: string;
  data: string;
  member_count: number;
  joined_member_count: number;
  max_length_message: number;
  created_at: number;
  custom_type: string;
  is_distinct: boolean;
  is_super: boolean;
  is_broadcast: boolean;
  is_public: boolean;
  is_discoverable: boolean;
  freeze: boolean;
  is_ephemeral: boolean;
  unread_message_count: number;
  unread_mention_count: number;
  ignore_profanity_filter: boolean;
  channel: Channel2;
  count_preference: string;
  created_by: CreatedBy;
  disappearing_message: DisappearingMessage;
  is_access_code_required: boolean;
  is_exclusive: boolean;
  is_muted: boolean;
  is_push_enabled: boolean;
  member_state: string;
  message_survival_seconds: number;
  metadata: any;
  my_role: string;
  push_trigger_option: string;
  read_receipt: ReadReceipt;
  sms_fallback: SmsFallback;
  ts_message_offset: number;
  user_last_read: any;
  inviter: Inviter;
  invited_at: any;
  is_hidden: boolean;
  hidden_state: string;
  last_message: LastMessage;
  members: Member[];
  joined_ts: number;
  last_queried_message?: any;
}

export interface ChannelsResponse {
  channels: Channel[];
  next?: string;
  ts: number;
}

export interface GetChannelsInput {
  token?: string;
  hidden_mode: string;
  show_empty: boolean;
  show_frozen: boolean;
  show_metadata: boolean;
  super_mode: string;
  unread_filter: string;
  user_id: string;
  order: string;
  show_member: boolean;
  public_mode: string;
  show_delivery_receipt: boolean;
  member_state_filter: string;
  limit: number;
  next?: string;
  distinct_mode: string;
  show_read_receipt: boolean;
}
