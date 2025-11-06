// me
import {FakeHingeAPI} from './fakeHinge';
import {HingeAPIImpl} from './hinge';

export const authToken = 'clcTyWvs_s277O4ZQIh-edPy5qKG0Mz6TqlEkh62LT4=';
export const sessionId = '1CBD1215-F431-4A3D-8B24-70179D11A58B';
export const myName = 'Ash';
export const myId = '3711734296147396244';

// aqib
// export const authToken = 'WX392211wCMLGjSnhdcO1vkk78yUZF5Yz9cnliAZyjU=';
// export const myName = 'Aqib';

const realHinge = new HingeAPIImpl(authToken, {
  'Host': 'prod-api.hingeaws.net',
  'user-agent': '/Hinge/11676 CFNetwork/3860.400.51 Darwin/25.3.0',
  'x-session-id': sessionId,
  'x-device-model': 'unknown',
  'x-app-identifier': 'co.hinge.mobile.ios',
  'x-device-id': '8EF20357-2329-4294-BD2B-33060D52F53A',
  'x-os-version': '26.3',
  'x-build-number': '11676',
  'priority': 'u=3',
  'x-app-version': '9.110.0',
  'x-device-platform': 'iOS',
  'accept-language': 'en',
  'x-device-region': 'US',
  'accept': '*/*',
  'content-type': 'application/json',
  'x-device-model-code': 'iPhone16,1',
  'x-install-id': 'F943AC32-EB7B-4B32-8BB8-AC8CA6F154E5',
  'accept-encoding': 'gzip, deflate, br',
}, myId);

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
