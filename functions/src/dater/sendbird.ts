import {
  GetMessagesParams,
  HingeMessagesResponse,
  SendBirdMessagesResponse,
} from './types';
import axios from 'axios';
import {ChannelsResponse, GetChannelsInput} from './sendbirdtypes';

const appId = '3cdad91c-1e0d-4a0d-bbee-9671988bf9e9';

export class SendbirdAPI {
  private host = `api-${appId}.sendbird.com`;
  private baseURL = `https://${this.host}/v3`;

  constructor(private sessionKey: string) {}

  async getMessages(
    channelId: string,
    params: GetMessagesParams
  ): Promise<HingeMessagesResponse> {
    const headers = this.getHeaders();
    const url = `${this.baseURL}/group_channels/${channelId}/messages?custom_types=%2A`;

    const response = await axios.get<HingeMessagesResponse>(url, {
      headers,
      params,
    });
    return response.data;
  }

  async getChannels(
    userId: string,
    input: GetChannelsInput
  ): Promise<ChannelsResponse> {
    const headers = this.getHeaders();
    const url = `${this.baseURL}/users/${userId}/my_group_channels`;

    const response = await axios.get(url, {headers, params: input});
    return response.data;
  }

  async sendMessage(
    channelId: string,
    myId: string,
    message: string
  ): Promise<SendBirdMessagesResponse> {
    const headers = this.getHeaders();
    const url = `${this.baseURL}/group_channels/${channelId}/messages`;

    return (
      await axios.post<SendBirdMessagesResponse>(
        url,
        {
          message,
          message_type: 'MESG',
          user_id: myId,
        },
        {headers}
      )
    ).data;
  }

  async deleteMessage(
    channelId: string,
    messageId: number
  ): Promise<SendBirdMessagesResponse> {
    const headers = this.getHeaders();
    const url = `${this.baseURL}/group_channels/${channelId}/messages/${messageId}`;

    return (await axios.delete<SendBirdMessagesResponse>(url, {headers})).data;
  }

  private getHeaders(): any {
    return {
      Host: this.host,
      accept: 'application/json',
      'session-key': this.sessionKey,
      sendbird: 'iOS,16.2,4.1.5,3CDAD91C-1E0D-4A0D-BBEE-9671988BF9E9',
      'request-sent-timestamp': '1675663138271',
      'user-agent': 'Jios/4.1.5',
      'sb-user-agent': 'iOS/c4.1.5',
      'accept-language': 'en-US,en;q=0.9',
    };
  }
}
