// me
import {FakeHingeAPI} from './fakeHinge';
import {HingeAPIImpl} from './hinge';

export const authToken = '5q7Z5nt0q1FI0vmH9-4699HqUdoJ46BVK-HE7H6137A=';
export const sessionId = 'E26E9D95-1B44-47A2-BC14-50C9455B9C4E';
export const myName = 'Ash';
export const myId = '2922149065963603846';

// aqib
// export const authToken = 'WX392211wCMLGjSnhdcO1vkk78yUZF5Yz9cnliAZyjU=';
// export const myName = 'Aqib';

const realHinge = new HingeAPIImpl(authToken, {
  // curl -H "Host: prod-api.hingeaws.net" -H "content-type: application/json" -H "x-device-platform: iOS" -H "user-agent: Hinge/11516 CFNetwork/3826.600.41 Darwin/24.6.0" -H "authorization: Bearer 5q7Z5nt0q1FI0vmH9-4699HqUdoJ46BVK-HE7H6137A=" -H "x-session-id: 4A487406-1A16-40B9-B3A8-128EDF832AC4" -H "accept: */*" -H "x-device-model-code: iPhone16,1" -H "x-install-id: F943AC32-EB7B-4B32-8BB8-AC8CA6F154E5" -H "accept-language: en" -H "x-hinge-waf-token: " -H "x-build-number: 11516" -H "x-device-region: US" -H "x-device-id: 8EF20357-2329-4294-BD2B-33060D52F53A" -H "x-app-version: 9.44.0" -H "x-device-model: unknown" -H "x-os-version: 18.7" --compressed "https://prod-api.hingeaws.net/connection"
  'Host': 'prod-api.hingeaws.net',
  'user-agent': 'Hinge/11516 CFNetwork/3860.100.1 Darwin/25.0.0',
  'x-session-id': sessionId,
  'x-device-model': 'unknown',
  'x-device-id': '8EF20357-2329-4294-BD2B-33060D52F53A',
  'x-os-version': '26.0',
  'x-build-number': '11516',
  'priority': 'u=3',
  'x-app-version': '9.44.0',
  'x-device-platform': 'iOS',
  'accept-language': 'en',
  'x-hinge-waf-token': '',
  'x-device-region': 'US',
  'accept': '*/*',
  'content-type': 'application/json',
  'x-device-model-code': 'iPhone16,1',
  'x-install-id': 'F943AC32-EB7B-4B32-8BB8-AC8CA6F154E5',
});

const testHinge = new FakeHingeAPI(authToken, {
  'x-session-id': 'CBBE87B8-BE72-497B-965D-63991CF00805',
  'x-device-model': 'unknown',
  'x-device-id': '8EF20357-2329-4294-BD2B-33060D52F53A',
  'x-os-version': '18.0',
  'x-build-number': '11516',
  'x-app-version': '9.44.0',
  'x-device-platform': 'iOS',
  'x-hinge-waf-token': '',
  'x-device-region': 'US',
  accept: '*/*',
  'content-type': 'application/json',
  'x-device-model-code': 'iPhone16,1',
  'x-install-id': '7E9F54FF-0149-4031-8402-25CF0A68492B',
  'accept-encoding': 'gzip, deflate, br',
});

const useFakes = false;
export const hinge = useFakes ? testHinge : realHinge;
