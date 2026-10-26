// Rebuild the offline PulseLab scene. Dependencies are build-time only.
// node tools/build_pulse_scene.mjs --deps PATH_TO_NODE_MODULES
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import fs from 'node:fs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const index=process.argv.indexOf('--deps');
if(index<0||!process.argv[index+1])throw new Error('Specify --deps pointing to node_modules containing three 0.180.0 and esbuild');
const deps=path.resolve(process.argv[index+1]);
const require=createRequire(path.join(deps,'_arraylab_build.cjs'));
let esbuild;
try{esbuild=require('esbuild');}catch{esbuild=createRequire(require.resolve('vite/package.json'))('esbuild');}
const threePackage=path.join(deps,'three','package.json');
if(JSON.parse(fs.readFileSync(threePackage,'utf8')).version!=='0.180.0')throw new Error('Use three 0.180.0 to preserve the original renderer');
const resource=path.join(root,'arraylab/features/pulse/resources/scene');
await esbuild.build({entryPoints:[path.join(resource,'src/entry.ts')],outfile:path.join(resource,'scene.bundle.js'),
  bundle:true,minify:true,format:'iife',platform:'browser',target:['chrome120'],nodePaths:[deps],sourcemap:false,legalComments:'none'});
fs.copyFileSync(path.join(deps,'three','LICENSE'),path.join(resource,'THREE-LICENSE.txt'));
console.log('Built offline PulseLab scene with Three.js 0.180.0');
