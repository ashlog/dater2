import * as fs from 'fs';
import {llmSort} from './dater';

async function test() {
  // read connection.json
  let connections: any[] = JSON.parse(
    await fs.promises.readFile(
      '/Users/ash/projects/dater/functions/src/connection.json',
      'utf8'
    )
  );

  connections = connections.filter(
    it =>
      it?.initiatorId !== '2922149065963603846' &&
      it.sentContent[0]?.comment &&
      it.sentContent[0].prompt?.question &&
      it.sentContent[0].prompt.answer
  );
  // connections = connections.sort(() => Math.random() - 0.5);
  // connections = connections.slice(0, 100);

  const pl = connections.map(
    it =>
      `${it.sentContent[0].prompt.question}: ${it.sentContent[0].prompt.answer} -- ${it.sentContent[0].comment}`
  );

  const newSort = await llmSort(pl);
  console.log(newSort);
}

test();
