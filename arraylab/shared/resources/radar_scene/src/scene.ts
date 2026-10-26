// Adapted from PulseLab 1.0.1 web/src/scene.ts. Rendering only; no signal solver.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

type Dict = Record<string, any>;
type Grid = {x_m:number[];y_m:number[];z_m:(number|null)[][];valid?:boolean[][]};
export type CameraState = {position:number[];target:number[];zoom:number};

function xyz(p:number[]):THREE.Vector3 {return new THREE.Vector3(p[0],p[2],-p[1]);}
function vector(p:any,name='坐标'):number[]{
  if(!Array.isArray(p)||p.length!==3||!p.every(Number.isFinite))throw new Error(name+'须包含3个有限数值');
  return p;
}
function pose(obj:Dict,time:number):number[]{
  const keys=(obj.trajectory||[]).slice().sort((a:Dict,b:Dict)=>a.time_s-b.time_s);
  if(keys.length>=2){let i=0;while(i<keys.length-2&&time>keys[i+1].time_s)i++;const a=keys[i],b=keys[i+1];
    const fraction=(time-a.time_s)/(b.time_s-a.time_s);return a.position_m.map((v:number,j:number)=>v+(b.position_m[j]-v)*fraction);}
  const base=keys.length?keys[0].position_m:obj.position_m,t0=keys.length?keys[0].time_s:0;
  return base.map((v:number,i:number)=>v+(obj.velocity_mps?.[i]||0)*(time-t0));
}
function attitudeAt(obj:Dict,time:number):number[]{
  // Follow the numerical model's per-keyframe default, including missing angles.
  const keys=(obj.trajectory||[]).slice().sort((a:Dict,b:Dict)=>a.time_s-b.time_s),fallback=obj.attitude_deg||[0,0,0];
  if(!keys.length)return fallback;if(keys.length===1)return keys[0].attitude_deg||fallback;
  let i=0;while(i<keys.length-2&&time>keys[i+1].time_s)i++;
  const a=keys[i],b=keys[i+1],fraction=(time-a.time_s)/(b.time_s-a.time_s),aa=a.attitude_deg||fallback,bb=b.attitude_deg||fallback;
  return aa.map((v:number,j:number)=>v+(((bb[j]-v+180)%360+360)%360-180)*fraction);
}
function groundHeight(point:number[],grid:Grid|null):number|null{
  if(!grid)return null;const xs=grid.x_m,ys=grid.y_m,x=point[0],y=point[1];
  if(x<xs[0]||x>xs[xs.length-1]||y<ys[0]||y>ys[ys.length-1])return null;
  const lower=(axis:number[],v:number)=>{let lo=0,hi=axis.length-1;while(hi-lo>1){const m=Math.floor((lo+hi)/2);if(axis[m]<=v)lo=m;else hi=m;}return Math.min(lo,axis.length-2);};
  const i=lower(xs,x),j=lower(ys,y),fx=(x-xs[i])/(xs[i+1]-xs[i]),fy=(y-ys[j])/(ys[j+1]-ys[j]);
  const cells=[[j,i],[j,i+1],[j+1,i],[j+1,i+1]];
  if(cells.some(([iy,ix])=>grid.z_m[iy]?.[ix]==null||!Number.isFinite(grid.z_m[iy][ix])||!(grid.valid?.[iy]?.[ix]??true)))return null;
  const [a,b,c,d]=cells.map(([iy,ix])=>Number(grid.z_m[iy][ix]));return a*(1-fx)*(1-fy)+b*fx*(1-fy)+c*(1-fx)*fy+d*fx*fy;
}
function labelSprite(text:string,color:string):THREE.Sprite{
  const canvas=document.createElement('canvas');canvas.width=512;canvas.height=96;const ctx=canvas.getContext('2d')!;
  ctx.fillStyle='rgba(13,31,46,.84)';ctx.beginPath();ctx.roundRect(3,3,506,85,18);ctx.fill();ctx.fillStyle=color;
  ctx.font='500 32px "Segoe UI", "Microsoft YaHei", sans-serif';ctx.textAlign='center';ctx.fillText(text,256,58,490);
  const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(canvas),transparent:true,depthTest:false}));return sprite;
}

export class RadarScene{
  private renderer:THREE.WebGLRenderer;private scene=new THREE.Scene();private camera:THREE.PerspectiveCamera;
  private controls:OrbitControls;private group=new THREE.Group();private ro:ResizeObserver;private frame=0;
  private actors:{mesh:THREE.Group;object:Dict}[]=[];private project:Dict={};private span=1000;private heightGrid:Grid|null=null;
  private beam?:THREE.Mesh;private pointingAxis?:THREE.Group;private running=false;private visible=true;private time=0;private last=0;private maxTime=5;private disposed=false;
  private renderCount=0;private terrainTriangles=0;private unknownActors=0;private caption='输入场景';
  onTime?:(time:number)=>void;
  constructor(private container:HTMLElement){
    this.renderer=new THREE.WebGLRenderer({antialias:true,alpha:false,preserveDrawingBuffer:true});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));this.renderer.setClearColor(0x0c1c2c);this.renderer.outputColorSpace=THREE.SRGBColorSpace;
    container.append(this.renderer.domElement);this.camera=new THREE.PerspectiveCamera(42,1,.1,1e8);
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.enableDamping=true;this.controls.dampingFactor=.08;
    this.controls.addEventListener('change',this.requestFrame);
    this.scene.add(new THREE.HemisphereLight(0xc9e6f2,0x1c3941,2.4));const sun=new THREE.DirectionalLight(0xffffff,2);sun.position.set(1,3,1);this.scene.add(sun,this.group);
    this.ro=new ResizeObserver(()=>this.resize());this.ro.observe(container);this.resize();
  }
  private resize(){if(this.disposed)return;const width=this.container.clientWidth||700,height=this.container.clientHeight||440;
    this.renderer.setSize(width,height,false);this.camera.aspect=width/height;this.camera.updateProjectionMatrix();this.requestFrame();}
  private requestFrame=()=>{if(!this.disposed&&this.visible&&!this.frame)this.frame=requestAnimationFrame(this.animate);};
  private animate=(now:number)=>{this.frame=0;if(this.disposed||!this.visible)return;
    if(this.running&&this.last)this.seek((this.time+(now-this.last)/1000)%this.maxTime,false);
    this.last=now;this.controls.update();this.renderer.render(this.scene,this.camera);this.renderCount++;
    if(this.running)this.requestFrame();else this.last=0;
  };
  private clearGroup(){for(const child of [...this.group.children]){child.traverse(obj=>{const mesh=obj as THREE.Mesh;mesh.geometry?.dispose();
    for(const material of (Array.isArray(mesh.material)?mesh.material:[mesh.material])){if(!material)continue;(material as THREE.MeshBasicMaterial).map?.dispose();material.dispose();}});
    this.group.remove(child);}this.actors=[];this.beam=undefined;this.pointingAxis=undefined;this.terrainTriangles=0;}
  setProject(project:Dict,grid:Grid|null=null,resetCamera=false,maxTime=5,caption='输入场景'){
    if(!project?.scene?.platform||!Array.isArray(project.scene.targets)||!project.array)throw new Error('场景配置缺少平台、目标或阵列');
    vector(project.scene.platform.position_m);project.scene.targets.forEach((item:Dict)=>vector(item.position_m));
    if(!Number.isFinite(maxTime)||maxTime<=0)throw new Error('预览时长须大于0');
    this.project=project;this.caption=caption;this.maxTime=maxTime;const terrain=project.scene.terrain;
    this.heightGrid=terrain?.enabled&&terrain.kind==='dem'&&terrain.x_m&&terrain.y_m&&terrain.heights_m?
      {x_m:terrain.x_m,y_m:terrain.y_m,z_m:terrain.heights_m,valid:terrain.valid}:grid;
    this.clearGroup();const points=[project.scene.platform.position_m,...project.scene.targets.map((o:Dict)=>o.position_m)];
    const requestedMin=project.scene.display_min_span_m,minSpan=typeof requestedMin==='number'&&Number.isFinite(requestedMin)&&requestedMin>0?requestedMin:100;
    this.span=Math.max(minSpan,...points.flat().map((n:number)=>Math.abs(n)),grid?Math.max(...grid.x_m.map(Math.abs),...grid.y_m.map(Math.abs)):0)*1.2;
    this.camera.near=Math.max(.01,this.span/1e5);this.camera.far=this.span*100;this.camera.updateProjectionMatrix();
    const helper=new THREE.GridHelper(this.span*2,20,0x2b5364,0x1d3748);helper.position.set(this.span*.4,-2,0);this.group.add(helper);
    const axes=new THREE.AxesHelper(this.span*.15);axes.position.set(0,1,0);this.group.add(axes);if(grid)this.addTerrain(grid);
    const platform=this.addActor(project.scene.platform,'雷达平台',0x63d5cf,true);this.actors.push({mesh:platform,object:project.scene.platform});
    project.scene.targets.forEach((obj:Dict,i:number)=>{const mesh=this.addActor(obj,obj.id||`目标 ${i+1}`,0xf5bf68,false);this.actors.push({mesh,object:obj});});
    if(project.array.beam_visible!==false){
      const length=this.span*.6,bw=Number(project.array.beamwidth_deg?.[0]||30),radius=length*Math.tan(Math.min(80,bw/2)*Math.PI/180);
      const geometry=new THREE.ConeGeometry(radius,length,40,1,true);geometry.translate(0,-length/2,0);geometry.rotateZ(Math.PI/2);
      this.beam=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({color:0x56ccbc,transparent:true,opacity:.085,side:THREE.DoubleSide,depthWrite:false}));platform.add(this.beam);
    }
    if(project.array.pointing_axis===true){
      // The group carries scan/pointing rotation; ArrowHelper keeps its own +X orientation.
      this.pointingAxis=new THREE.Group();this.pointingAxis.add(new THREE.ArrowHelper(
        new THREE.Vector3(1,0,0),new THREE.Vector3(),this.span*.6,0x63d5cf,this.span*.045,this.span*.022));
      platform.add(this.pointingAxis);
    }
    this.seek(this.time);if(resetCamera)this.resetCamera();else this.requestFrame();
  }
  private addTerrain(grid:Grid){
    const nx=grid.x_m.length,ny=grid.y_m.length,stride=Math.max(1,Math.ceil(Math.max(nx,ny)/180)),xs:number[]=[],ys:number[]=[];
    for(let i=0;i<nx;i+=stride)xs.push(i);if(xs[xs.length-1]!==nx-1)xs.push(nx-1);
    for(let i=0;i<ny;i+=stride)ys.push(i);if(ys[ys.length-1]!==ny-1)ys.push(ny-1);
    const positions:number[]=[],colors:number[]=[],valid:boolean[]=[],indices:number[]=[];
    const heights=ys.flatMap(j=>xs.map(i=>grid.z_m[j]?.[i]).filter((z):z is number=>z!==null&&Number.isFinite(z)));
    const min=Math.min(0,...heights),max=Math.max(1,...heights);
    ys.forEach(j=>xs.forEach(i=>{const z=grid.z_m[j]?.[i],ok=Number.isFinite(z)&&z!==null&&(grid.valid?.[j]?.[i]??true);valid.push(ok);
      positions.push(grid.x_m[i],ok?Number(z):0,-grid.y_m[j]);const f=ok?(Number(z)-min)/(max-min):0,c=new THREE.Color().setHSL(.48-f*.18,.23,.2+f*.22);colors.push(c.r,c.g,c.b);}));
    for(let j=0;j<ys.length-1;j++)for(let i=0;i<xs.length-1;i++){const a=j*xs.length+i,b=a+1,c=a+xs.length,d=c+1;if(valid[a]&&valid[b]&&valid[c])indices.push(a,c,b);if(valid[b]&&valid[c]&&valid[d])indices.push(b,c,d);}
    const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));geometry.setIndex(indices);geometry.computeVertexNormals();
    this.terrainTriangles=indices.length/3;this.group.add(new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({vertexColors:true,side:THREE.DoubleSide,roughness:1,metalness:0})));
  }
  private worldPosition(obj:Dict,time:number):number[]|null{
    const position=pose(obj,time);if(obj.height_mode==='agl'){const height=groundHeight(position,this.heightGrid);if(height===null)return null;position[2]+=height;}return position;
  }
  private addActor(obj:Dict,label:string,color:number,radar:boolean){
    const actor=new THREE.Group(),size=this.span*.013;
    actor.add(new THREE.Mesh(radar?new THREE.BoxGeometry(size*1.4,size,size):new THREE.OctahedronGeometry(size*.55),new THREE.MeshStandardMaterial({color,metalness:.2,roughness:.6})));
    if(radar){const dish=new THREE.Mesh(new THREE.BoxGeometry(size*.15,size*1.8,size*1.5),new THREE.MeshStandardMaterial({color:0xd4eeee}));dish.position.set(size*.6,size*.6,0);actor.add(dish);}
    const labelMesh=labelSprite(label,radar?'#80e0d8':'#f5cf8d');labelMesh.position.y=size*2.8;labelMesh.scale.set(size*10,size*1.9,1);actor.add(labelMesh);
    const times=obj.trajectory?.length>=2?obj.trajectory.map((key:Dict)=>key.time_s):[0,this.maxTime];
    // Preserve original dashed tracks, adding terrain height for AGL and leaving unknown segments blank.
    const positions=times.map((t:number)=>this.worldPosition(obj,t)),segments:THREE.Vector3[]=[];
    for(let i=1;i<positions.length;i++)if(positions[i-1]&&positions[i]){const a=xyz(positions[i-1]),b=xyz(positions[i]);if(a.distanceTo(b)>1)segments.push(a,b);}
    if(segments.length){const line=new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(segments),new THREE.LineDashedMaterial({color,transparent:true,opacity:.6,dashSize:this.span*.008,gapSize:this.span*.005}));line.computeLineDistances();this.group.add(line);}
    this.group.add(actor);return actor;
  }
  seek(t:number,request=true){
    if(!Number.isFinite(t)||t<0)throw new Error('场景预览时间须为非负有限数值');this.time=t;this.unknownActors=0;
    for(const actor of this.actors){const position=this.worldPosition(actor.object,t);actor.mesh.visible=position!==null;
      if(!position){this.unknownActors++;continue;}actor.mesh.position.copy(xyz(position));const attitude=attitudeAt(actor.object,t);
      actor.mesh.rotation.set((attitude[2]||0)*Math.PI/180,(attitude[0]||0)*Math.PI/180,(attitude[1]||0)*Math.PI/180,'YZX');}
    if(this.beam||this.pointingAxis){const a=this.project.array,az=(Number(a.pointing_deg?.[0]||0)+Number(a.scan_rate_deg_s?.[0]||0)*t)*Math.PI/180,
      el=(Number(a.pointing_deg?.[1]||0)+Number(a.scan_rate_deg_s?.[1]||0)*t)*Math.PI/180;
      this.beam?.rotation.set(0,az,el,'YZX');this.pointingAxis?.rotation.set(0,az,el,'YZX');}
    this.onTime?.(t);if(request)this.requestFrame();
  }
  play(value:boolean){this.running=!!value;this.last=0;this.requestFrame();}
  setVisible(value:boolean){this.visible=!!value;this.last=0;if(this.visible)this.requestFrame();else if(this.frame){cancelAnimationFrame(this.frame);this.frame=0;}}
  private applyCamera(state:CameraState){
    // Consume pending orbit/pan/zoom deltas before setting an exact saved pose.
    // Otherwise the next animation frame reapplies damping from the last drag.
    const damping=this.controls.enableDamping;this.controls.enableDamping=false;
    try{this.controls.update();this.camera.position.fromArray(state.position);this.controls.target.fromArray(state.target);this.camera.zoom=state.zoom;
      this.camera.updateProjectionMatrix();this.controls.update();}
    finally{this.controls.enableDamping=damping;}
    this.requestFrame();
  }
  resetCamera(){this.applyCamera({position:[this.span*.85,this.span*.65,this.span*.85],target:[this.span*.4,0,0],zoom:1});}
  restoreCamera(state:CameraState){vector(state.position,'相机位置');vector(state.target,'相机目标');if(!Number.isFinite(state.zoom)||state.zoom<=0)throw new Error('相机缩放无效');
    this.applyCamera(state);}
  displayStatus(){return {caption:this.caption,unknown_actors:this.unknownActors};}
  getState(){let meshes=0;this.group.traverse(obj=>{if((obj as THREE.Mesh).isMesh)meshes++;});return {time_s:this.time,playing:this.running,visible:this.visible,render_count:this.renderCount,
    mesh_count:meshes,terrain_triangles:this.terrainTriangles,unknown_actors:this.unknownActors,caption:this.caption,max_time_s:this.maxTime,
    camera:{position:this.camera.position.toArray(),target:this.controls.target.toArray(),zoom:this.camera.zoom},
    actors:this.actors.map(a=>({id:a.object.id||'平台',visible:a.mesh.visible,position:a.mesh.position.toArray(),quaternion:a.mesh.quaternion.toArray()})),
    beam_direction:this.beam?new THREE.Vector3(1,0,0).applyQuaternion(this.beam.getWorldQuaternion(new THREE.Quaternion())).toArray():null,
    pointing_direction:this.pointingAxis?new THREE.Vector3(1,0,0).applyQuaternion(this.pointingAxis.getWorldQuaternion(new THREE.Quaternion())).toArray():null,
    geometries:this.renderer.info.memory.geometries,textures:this.renderer.info.memory.textures};}
  png(){if(this.disposed)throw new Error('场景已关闭');this.controls.update();this.renderer.render(this.scene,this.camera);return this.renderer.domElement.toDataURL('image/png');}
  dispose(){if(this.disposed)return;this.disposed=true;this.running=false;cancelAnimationFrame(this.frame);this.frame=0;this.ro.disconnect();this.controls.removeEventListener('change',this.requestFrame);this.controls.dispose();this.clearGroup();this.renderer.dispose();this.renderer.forceContextLoss();this.renderer.domElement.remove();}
}
