import { RadarScene } from './scene';

const container=document.querySelector<HTMLElement>('#scene')!;
const statusElement=document.querySelector<HTMLElement>('#scene-status')!;
let scene:RadarScene|null=null,error='',supported=false,disposed=false,hasScene=false;
function status(){return {ready:true,supported,error,disposed,...(scene?.getState()||{})};}
function call(action:()=>void){try{if(!scene||disposed)throw new Error(error||'三维场景尚不可用');action();error='';return status();}
  catch(e){error=e instanceof Error?e.message:String(e);statusElement.textContent=error;return status();}}
const api={ready:false,status,getState:status,
  setScene:(data:any)=>call(()=>{scene!.setProject(data.project,data.grid??null,!hasScene||!!data.reset_camera,data.max_time_s??5,data.caption||'输入场景');scene!.seek(data.time_s??0);hasScene=true;}),
  setTime:(seconds:number)=>call(()=>scene!.seek(seconds)),
  setPlaying:(value:boolean)=>call(()=>scene!.play(value)),
  setVisible:(value:boolean)=>call(()=>scene!.setVisible(value)),
  resetCamera:()=>call(()=>scene!.resetCamera()),
  restoreCamera:(camera:any)=>call(()=>scene!.restoreCamera(camera)),
  png:()=>{if(!scene||disposed)return '';try{return scene.png();}catch(e){error=String(e);return '';}},
  dispose:()=>{scene?.dispose();disposed=true;return status();}
};
(window as any).ArrayLabScene=api;
try{scene=new RadarScene(container);supported=true;
  scene.onTime=time=>{const state=scene!.displayStatus();statusElement.textContent=`${state.caption} · ${time.toFixed(2)} s · 预览不重算回波${state.unknown_actors?' · '+state.unknown_actors+'个离地高度位置未知':''}`;};
  statusElement.textContent='等待场景输入；拖动旋转 · 滚轮缩放 · 右键平移';
}catch(e){error=e instanceof Error?e.message:String(e);statusElement.textContent='WebGL 不可用：'+error;}
api.ready=true;
document.addEventListener('visibilitychange',()=>scene?.setVisible(!document.hidden));
window.addEventListener('pagehide',()=>api.dispose());
