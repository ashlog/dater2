import {Channel, GetChannelsInput} from './sendbirdtypes';
import {SendbirdAPI} from './sendbird';
import {HingeAPI} from './hinge';
import {qna} from './qna';
import {chat, CHAT_PREPROMPT} from './openai';
import {HingeMessagesResponse} from './types';
import {dateDifferenceFormatted, timeout} from './utils';
import {authToken, myName} from './token';
import {hinge} from "../functions/src/dater/token";

const sendbirdSession = '4bd6406e9736c5f0b3df941926bc391f88f7ebe7';
const myUserId = '2922149065963603846';

const getChannelsInput: GetChannelsInput = {
  hidden_mode: 'unhidden_only',
  show_empty: true,
  show_frozen: true,
  show_metadata: true,
  super_mode: 'all',
  unread_filter: 'all',
  user_id: myUserId,
  order: 'chronological',
  show_member: true,
  public_mode: 'all',
  show_delivery_receipt: true,
  member_state_filter: 'all',
  limit: 100,
  distinct_mode: 'all',
  show_read_receipt: true,
};

const sendbird = new SendbirdAPI(sendbirdSession);

async function getLastFewMessages(
  channel: Channel
): Promise<HingeMessagesResponse> {
  const prevLimit = 5;
  // get all chats formatted as a string in the format "Me:" and "Them:"
  const messages = await sendbird.getMessages(channel.channel_url, {
    next: '',
    message_ts: Date.now(),
    include_reactions: true,
    include: false,
    with_sorted_meta_array: false,
    prev_limit: prevLimit,
    next_limit: 0,
    reverse: true,
    show_subchannel_messages_only: false,
    include_reply_type: 'all',
    include_parent_message_info: false,
    include_thread_info: false,
    is_sdk: true,
    include_poll_details: true,
  });
  return messages;
}

const bannedUsers = new Set([
  '2426806731599250743',
  '2680390216870528648', //Jordan
  '2414306284498060822', //Emily
  '2773036911793538913', //Onaiza
  '2388402992358360760', //Julia White
  '2886219684040934937', //Katie
  '3103412004853909058', //Crystal
  '2856957496554161276', //Valentina
]);

const blacklist = new Set([
  'Danielle',
  'Maya',
  'Erin',
  'Kay',
  'Bree',
  'Maya',
  'Kathryn',
  'Crystal',
  'Liza',
]);

const whitelist = new Set([
  'Ivana',
  'Katie',
  'Holly',
  'Allie',
  'Nisa',
  'Kristin',
  'Eve',
  'Caitlin',
  'Salma',
  'Grace',
  'Xiatong',
  'Tamica',
  'Manuela',
  'Miri',
  'Linda',
  'Aya',
  'Yuki',
  'Nicole',
  'Cassie',
  'Charity',
  'Ale',
  'Malia',
  'Lejla',
  'X',
  'Elisa',
  'Leonie',
  'Ahh',
  'Daisy',
  'Maci',
  'Maya',
  'Bridgett',
  'Rachel',
  'Gina',
  'Nicole',
  'Stacy',
  'miranda',
  'Teresa',
  'Carolyn',
  'Lexie',
]);

async function reply(channel: Channel) {
  const lastMessage = channel.last_message;
  const memberThem = channel.members.filter(it => it.user_id !== myUserId)[0];

  if (bannedUsers.has(memberThem.user_id)) return;

  // const readMyLastMessage =
  //   memberThem.user_id &&
  //   channel.read_receipt[memberThem.user_id] &&
  //   channel.read_receipt[memberThem.user_id] === lastMessage.created_at &&
  //   lastMessage.user.user_id === myUserId;

  const allProfiles = await hinge.getProfiles([memberThem.user_id]);
  if (allProfiles.length === 0) return;
  const profile = allProfiles[0].profile;
  const theirName = profile.firstName || profile.lastName || 'Them';

  //blacklist
  if (blacklist.has(theirName)) {
    console.log('BAN THIS USER', profile.userId, theirName);
    return;
  }

  let deletedLast = false;

  //whitelist
  // if (!whitelist.has(theirName)) return;

  //redo and delete last message
  // if (theirName === 'name') {
  //   const messages = await getLastFewMessages(channel);
  //   //delete the last message
  //   const lastMessage = messages.messages[0];
  //   if (lastMessage.user.user_id === myUserId) {
  //     console.log(
  //       'Deleting last message',
  //       `${lastMessage.user.nickname}: ${lastMessage.message}`
  //     );
  //     const sendBirdMessagesResponse = await sendbird.deleteMessage(
  //       channel.channel_url,
  //       lastMessage.message_id
  //     );
  //     console.log(sendBirdMessagesResponse);
  //     return;
  //   }
  // }

  const messages = await getLastFewMessages(channel);
  // resend a new message
  if (
    (messages.messages.length === 1 &&
      (messages.messages[0].message === "Hey there! How's it going?" ||
        messages.messages[0].message === '')) ||
    messages.messages.length === 0
  ) {
    //delete the last message
    const lastMessage = messages.messages[0];
    if (lastMessage && lastMessage.user.user_id === myUserId) {
      console.log(
        'Deleting last message',
        `${lastMessage.user.nickname}: ${lastMessage.message}`
      );
      const sendBirdMessagesResponse = await sendbird.deleteMessage(
        channel.channel_url,
        lastMessage.message_id
      );
      console.log(sendBirdMessagesResponse);
      deletedLast = true;
      // return;
    }
  }

  if (
    !lastMessage ||
    lastMessage.user.user_id !== myUserId ||
    messages.messages.length === 0 ||
    deletedLast
  ) {
    const chat = await getChat(messages, channel, myUserId, theirName);
    // console.log("Read my last message:", readMyLastMessage);

    // get profile from hinge and add the questions they've answered into a string
    const qna = profile.answers
      .map(it => `${questionMap.get(it.questionId)}: ${it.response}`)
      .join('\n');
    const profilePrompt = getProfilePrompt(theirName, qna, chat);
    let profileResponse = await getProfileResponse(profilePrompt);
    await timeout(2000);
    console.log(profilePrompt);
    console.log('Response: ' + profileResponse + '\n----\n');
    profileResponse = profileResponse.split('"')[1];
    if (!profileResponse) {
      console.log('Extracted nothing, doing nothing');
      return;
    }
    console.log('Extracted: ' + profileResponse);

    // try {
    //   const messageRequestHandler = await sendbird.sendMessage(
    //     channel.channel_url,
    //     myUserId,
    //     profileResponse
    //   );
    //
    //   await hinge.sendMessage([
    //     {
    //       isMatchMessage: false,
    //       recipient: profile.userId,
    //       text: profileResponse,
    //       created: new Date(messageRequestHandler.created_at).toISOString(),
    //       messageId: messageRequestHandler.message_id.toString(),
    //     },
    //   ]);
    // } catch (e) {
    //   console.error(e);
    // }
  }
}

async function getChat(
  messages: HingeMessagesResponse,
  channel,
  myUserId,
  theirName
) {
  return messages.messages
    .reverse()
    .map(it => {
      const dateDiff = dateDifferenceFormatted(
        new Date(it.created_at),
        new Date()
      );
      const name = it.user.user_id === myUserId ? myName : theirName;
      return `${name}: (${dateDiff}) ${it.message}`;
    })
    .join('\n');
}

function getProfilePrompt(theirName: string, qna: string, chat: string) {
  let profilePrompt = hingeProfileText(theirName, qna);
  if (chat) {
    profilePrompt += `
[CHAT]
${chat}`;
  }
  profilePrompt += `\n\n[QUESTION] What should ${myName}'s flirtatious, but not awkward, next message should be? Keep it short and succinct. Ask questions about her if it is relevant. Wrap your answer in quotes. Do not generate anything else.`;
  return profilePrompt;
}

export function hingeProfileText(theirName: string, qna: string) {
  const todayDateString = new Date().toLocaleDateString('en-US', {
    weekday: 'long',
    month: 'long',
    day: 'numeric',
  });

  let profilePrompt = `My name is ${myName}. I like latte and tea dates. My hobbies include cooking, hiking, and photography. I am a software engineer. I live in Mountain View, CA but currently am out of state. Today is ${todayDateString}. Hint at setting up a date sometime between today and July 31th as I will going back to Mountain View. Use the format: [ANSWER ${myName}] "message"`;
  profilePrompt += `
====
Hinge Profile: Hadyn
Profile Questions:
The one thing you should know about me is: I do not do things. All of these pictures are photoshopped.
The hallmark of a good relationship is: a Morticia and Gomez level of passion and attraction.
My mantra is: You're not hardcore unless you live hardcore.

[CHAT]
Hadyn: What have you got in mind?

[QUESTION] What should ${myName}'s flirtatious, but not awkward, next message should be?

[INTUITION] The message should combine two of ${myName}'s hobbies, cooking and photography, while also referencing Hadyn's interests and sense of humor. It should be flirtatious without being too forward and should hopefully spark further conversation.
[ANSWER ${myName}] "Well, since you're into Morticia and Gomez level of passion and attraction, maybe we should plan a cooking date where we can whip up a wickedly delicious meal together. And who knows, maybe we'll end up recreating their famous tango dance afterward ;)"
====
`;
  profilePrompt += `Hinge Profile: ${theirName}
Profile Questions:
${qna}
`;
  return profilePrompt;
}

async function getProfileResponse(profilePrompt) {
  const profileResponse = await chat(profilePrompt, CHAT_PREPROMPT);
  return profileResponse;
}

async function run() {
  let next;
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const channels = await sendbird.getChannels(myUserId, {
      ...getChannelsInput,
      token: next,
    });
    next = channels.next;

    for (const channel of channels.channels) {
      await reply(channel);
    }
    if (!next) break;
  }
}

const questionMap = new Map<string, string>();
qna.prompts.forEach(prompt => {
  questionMap.set(prompt.id, prompt.prompt);
});
run();
