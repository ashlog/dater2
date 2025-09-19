import {
  loc_bayPalo,
  loc_lilburn,
  loc_monterey,
  loc_mtv,
  loc_san_jose,
  loc_santaCruz,
  loc_sf,
  run,
  loc_marrietta,
} from './dater';
import {logger} from 'firebase-functions';
import {onSchedule} from 'firebase-functions/v2/scheduler';
import {hinge, myId, myName} from './dater/token';
import fs from 'fs';
import path from 'path';
import {SendbirdAPI} from './dater/sendbird';
import {GetChannelsInput} from './dater/sendbirdtypes';
import {chatImage, getDemographics} from './dater/openai';
import extract from 'extract-json-from-string';

export const cron = onSchedule(
  {
    timeoutSeconds: 60,
    // Every hour 5-7pm
    schedule: '0 1 * * *',
    // Every 10 minutes
    // schedule: "*/10 * * * *",
    memory: '256MiB',
  },
  async event => {
    logger.info('Extracting.');
    await run([loc_bayPalo, loc_mtv], 10);
    logger.info('Done Extracting.');
  }
);

function main() {
  run(
    // [loc_san_jose, loc_mtv, loc_bayPalo, loc_sf, loc_santaCruz, loc_lilburn],
    // [loc_sf],
    [loc_bayPalo, loc_mtv],
    // [loc_santaCruz],
    // [loc_marrietta],
    // [loc_santaCruz],
    // [loc_lilburn],
    1000
  ).then(() => {
    console.log('done');
  });
}

async function updateLikeBuckets() {
  // await updateLikeBucketsJson();
}

async function updateLikeBucketsJson() {
  const likeBucket: string[] = [];
  const dislikeBucket: string[] = [];

  const sendbirdSession = '4bd6406e9736c5f0b3df941926bc391f88f7ebe7';
  const myUserId = '2922149065963603846';

  const allUsers: string[] = [];
  const sendbird = new SendbirdAPI(sendbirdSession);
  let next;
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const channels = await sendbird.getChannels(myUserId, {
      hidden_mode: 'all',
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
      token: next,
    });
    next = channels.next;

    for (const channel of channels.channels) {
      const memberThem = channel.members.filter(
        it => it.user_id !== myUserId
      )[0];
      if (memberThem) allUsers.push(memberThem.user_id);
    }
    if (!next) break;
  }

  for (const id of allUsers) {
    const profiles = await hinge.getProfiles([id]);
    if (profiles.length === 0) continue;
    const profile = profiles[0].profile;
    const images = profile.photos.map(it => it.url);

    for (const image of images) {
      const demographicsString = await getDemographics(image);
      const json = {
        image,
        ...extract(demographicsString)[0],
      };
      if (Object.entries(json).length !== 5) {
        console.log('Ignored', image);
        continue;
      }

      console.log(json);
      const weightOk = !['skinny', 'overweight', 'obese'].find(
        it => it === json.bodyThickness
      );
      if (weightOk) {
        likeBucket.push(image);
      } else {
        dislikeBucket.push(image);
      }

      await fs.promises.writeFile(
        '/Users/ash/projects/dater/functions/src/buckets/likes/bucket.json',
        JSON.stringify(likeBucket, null)
      );
      await fs.promises.writeFile(
        '/Users/ash/projects/dater/functions/src/buckets/dislikes/bucket.json',
        JSON.stringify(dislikeBucket, null)
      );
    }
  }
  console.log('Done');
}

// Test main: read user's recent chat messages and write to JSON
async function readChatMessages() {
  const session = '4bd6406e9736c5f0b3df941926bc391f88f7ebe7';
  const userId = myId;
  const perChannel = 5;
  const maxChannels = 25;

  const sendbird = new SendbirdAPI(session);

  let next: string | undefined = undefined;
  const channelsOut: any[] = [];
  let scanned = 0;

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const params: GetChannelsInput = {
      hidden_mode: 'all',
      show_empty: true,
      show_frozen: true,
      show_metadata: true,
      super_mode: 'all',
      unread_filter: 'all',
      user_id: userId,
      order: 'chronological',
      show_member: true,
      public_mode: 'all',
      show_delivery_receipt: true,
      member_state_filter: 'all',
      limit: Math.min(100, maxChannels - scanned),
      distinct_mode: 'all',
      show_read_receipt: true,
      token: next,
    };

    const channels = await sendbird.getChannels(userId, params);
    next = channels.next;

    for (const channel of channels.channels) {
      if (scanned >= maxChannels) break;

      const them = channel.members.find(m => m.user_id !== userId);
      const messagesResp = await sendbird.getMessages(channel.channel_url, {
        next: '',
        message_ts: Date.now(),
        include_reactions: true,
        include: false,
        with_sorted_meta_array: false,
        prev_limit: perChannel,
        next_limit: 0,
        reverse: true,
        show_subchannel_messages_only: false,
        include_reply_type: 'all',
        include_parent_message_info: false,
        include_thread_info: false,
        is_sdk: true,
        include_poll_details: true,
      });

      const messages = messagesResp.messages.map(m => ({
        message_id: m.message_id,
        text: m.message,
        created_at: m.created_at,
        sender: {
          user_id: m.user.user_id,
          nickname: m.user.nickname,
        },
      }));

      channelsOut.push({
        channel_url: channel.channel_url,
        last_message_ts: channel.last_message?.created_at ?? null,
        member_them: them ? {user_id: them.user_id, nickname: them.nickname} : null,
        messages,
      });
      scanned++;
    }

    if (!next || scanned >= maxChannels) break;
  }

  const outPath = path.join(__dirname, '../src/chat_messages.json');
  await fs.promises.writeFile(outPath, JSON.stringify({
    userId,
    count: channelsOut.length,
    channels: channelsOut,
  }, null, 2));
  console.log(`Wrote ${channelsOut.length} channels to ${outPath}`);
}

// async function test() {
//   // read connection.json
//   let connections: any[] = JSON.parse(
//     await fs.promises.readFile(
//       '/Users/ash/projects/dater/functions/src/connection.json',
//       'utf8'
//     )
//   );
//
//   connections = connections.sort(() => Math.random() - 0.5);
//   connections = connections.slice(0, 100);
//
//   const plConnections = connections.filter(
//     it =>
//       it?.sentContent[0]?.comment &&
//       it.sentContent[0].prompt?.question &&
//       it.sentContent[0].prompt.answer
//   );
//
//   const pl = plConnections.map(
//     it =>
//       `${it.sentContent[0].prompt.question}: ${it.sentContent[0].prompt.answer} -- ${it.sentContent[0].comment}`
//   );
//
//   const newSort = await llmSort(pl);
//   console.log(newSort);
// }
//
// test();

main();
// updateLikeBuckets();
// readChatMessages();
