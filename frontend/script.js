const video=document.getElementById("camera"), startButton=document.getElementById("startButton"), stopButton=document.getElementById("stopButton");
const placeholder=document.getElementById("placeholder"), cameraStatus=document.getElementById("cameraStatus"), message=document.getElementById("message");
const signResult=document.getElementById("signResult"), translationResult=document.getElementById("translationResult"), confidenceResult=document.getElementById("confidenceResult"), historyBody=document.getElementById("historyBody");
const overlay=document.getElementById("landmarkCanvas"), overlayCtx=overlay.getContext("2d");
const handToggle=document.getElementById("handToggle"), bodyToggle=document.getElementById("bodyToggle"), faceToggle=document.getElementById("faceToggle");
const replayVoiceButton=document.getElementById("replayVoiceButton");
const manipuriAudio=document.getElementById("manipuriAudio");
let stream=null,timer=null,busy=false,candidate="",count=0,lastSaved="",lastAutoSpoken="";
let currentAudioUrl="";
const sentenceItems=[];
const sentenceWords=document.getElementById("sentenceWords"),sentenceTranslation=document.getElementById("sentenceTranslation");
const sentenceMessage=document.getElementById("sentenceMessage"),saveSentenceButton=document.getElementById("saveSentenceButton");
const playSentenceButton=document.getElementById("playSentenceButton"),undoWordButton=document.getElementById("undoWordButton"),clearSentenceButton=document.getElementById("clearSentenceButton");
saveSentenceButton.onclick=saveSentence;playSentenceButton.onclick=playSentence;undoWordButton.onclick=()=>{sentenceItems.pop();renderSentence()};
clearSentenceButton.onclick=()=>{sentenceItems.length=0;renderSentence();sentenceMessage.textContent="Sentence cleared."};
function renderSentence(){
 sentenceWords.innerHTML="";
 if(!sentenceItems.length)sentenceWords.innerHTML='<span class="empty-sentence">Your sentence will appear here.</span>';
 sentenceItems.forEach(item=>{const chip=document.createElement("span");chip.className="word-chip";chip.textContent=item.translation||item.sign;sentenceWords.appendChild(chip)});
 sentenceTranslation.textContent=sentenceItems.map(x=>x.translation).filter(Boolean).join(" ")||"—";
 const has=sentenceItems.length>0;
 saveSentenceButton.disabled=!has;playSentenceButton.disabled=!has;undoWordButton.disabled=!has;clearSentenceButton.disabled=!has;
}
function addSentenceWord(data){
 if(sentenceItems.length&&sentenceItems[sentenceItems.length-1].sign===data.sign)return;
 sentenceItems.push({sign:data.sign,translation:data.translation||"",audio_url:data.audio_url||""});
 renderSentence();sentenceMessage.textContent="Added "+data.sign+" to sentence.";
}
async function saveSentence(){
 if(!sentenceItems.length)return;
 try{
  const response=await fetch("/api/sentences",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({signs:sentenceItems.map(x=>x.sign),translation:sentenceItems.map(x=>x.translation).join(" ")})});
  const data=await response.json();if(!response.ok)throw new Error(data.detail||"Save failed");
  sentenceMessage.textContent="Sentence saved successfully (ID "+data.id+").";
 }catch(e){sentenceMessage.textContent="Could not save sentence: "+e.message}
}
async function playSentence(){
 const clips=sentenceItems.map(x=>x.audio_url).filter(Boolean);
 if(!clips.length){sentenceMessage.textContent="No recorded audio clips are mapped to these words yet.";return}
 sentenceMessage.textContent="Playing sentence word by word...";
 for(const url of clips){
  try{await new Promise((resolve,reject)=>{const a=new Audio(url);a.onended=resolve;a.onerror=reject;a.play().catch(reject)})}
  catch(e){sentenceMessage.textContent="A word audio clip could not play. Check the audio mapping.";return}
 }
 sentenceMessage.textContent="Sentence audio finished.";
}

let latestLandmarks={hand:[],body:[],face:[]};
const canvas=document.createElement("canvas"),ctx=canvas.getContext("2d");
startButton.onclick=startCamera;stopButton.onclick=stopCamera;document.getElementById("refreshButton").onclick=loadHistory;
replayVoiceButton.onclick=playCurrentAudio;
[handToggle,bodyToggle,faceToggle].forEach(el=>el.addEventListener("change",drawLandmarks));
function playCurrentAudio(){
 if(!currentAudioUrl){
  message.textContent="No recorded Meiteilon audio is linked to this sign yet. Add the audio file and its mapping in sign_translations.json.";
  return;
 }
 manipuriAudio.currentTime=0;
 manipuriAudio.play().then(()=>{message.textContent="Playing Meiteilon translation."})
  .catch(()=>{message.textContent="Audio could not autoplay. Press Replay Meiteilon audio."});
}
function setTranslationAudio(url){
 url=url||"";
 if(url===currentAudioUrl)return;
 currentAudioUrl=url;
 replayVoiceButton.disabled=!currentAudioUrl;
 manipuriAudio.pause();
 manipuriAudio.removeAttribute("src");
 if(currentAudioUrl){manipuriAudio.src=currentAudioUrl;manipuriAudio.load();}
}
function resizeOverlay(){
 const rect=video.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
 overlay.width=Math.round(rect.width*dpr);overlay.height=Math.round(rect.height*dpr);
 overlayCtx.setTransform(dpr,0,0,dpr,0,0);
 drawLandmarks();
}
window.addEventListener("resize",resizeOverlay);
function drawLandmarks(){
 const rect=video.getBoundingClientRect(),w=rect.width,h=rect.height;
 overlayCtx.clearRect(0,0,w,h);
 // The preview is mirrored, so mirror normalized x coordinates in the overlay too.
 const groups=[
  {name:"hand",enabled:handToggle.checked,color:"#38e28b",radius:3.2},
  {name:"body",enabled:bodyToggle.checked,color:"#ff4fd8",radius:3.0},
  {name:"face",enabled:faceToggle.checked,color:"#ffe45c",radius:1.8}
 ];
 for(const group of groups){
  if(!group.enabled)continue;
  for(const p of (latestLandmarks[group.name]||[])){
   const x=p.x*w,y=p.y*h;
   overlayCtx.beginPath();overlayCtx.arc(x,y,group.radius,0,Math.PI*2);
   overlayCtx.fillStyle=group.color;overlayCtx.fill();
  }
 }
}
async function startCamera(){
 try{
  stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:"user",width:{ideal:640},height:{ideal:480}},audio:false});
  video.srcObject=stream;await video.play();placeholder.classList.add("hidden");resizeOverlay();
  cameraStatus.textContent="Camera running";message.textContent="Show a trained sign to the camera.";
  startButton.disabled=true;stopButton.disabled=false;
  // 70 ms target interval: faster than the original 180 ms while avoiding
  // sending multiple concurrent requests to the Python model.
  timer=setInterval(sendFrame,70);
 }catch(e){message.textContent="Camera access failed: "+e.message+". Allow camera permission in your browser."}
}
function stopCamera(){
 if(timer)clearInterval(timer);timer=null;if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;
 latestLandmarks={hand:[],body:[],face:[]};drawLandmarks();placeholder.classList.remove("hidden");
 cameraStatus.textContent="Camera stopped";message.textContent="Camera stopped.";
 startButton.disabled=false;stopButton.disabled=true;candidate="";count=0;lastSaved="";lastAutoSpoken="";setTranslationAudio("");
}
async function sendFrame(){
 if(busy||!stream||video.readyState<2)return;busy=true;
 try{
  canvas.width=video.videoWidth;canvas.height=video.videoHeight;ctx.drawImage(video,0,0,canvas.width,canvas.height);
  const blob=await new Promise(resolve=>canvas.toBlob(resolve,"image/jpeg",.68));if(!blob)return;
  const form=new FormData();form.append("frame",blob,"frame.jpg");form.append("detect_body",String(bodyToggle.checked));form.append("detect_face",String(faceToggle.checked));
  const response=await fetch("/api/predict",{method:"POST",body:form});const data=await response.json();
  if(!response.ok)throw new Error(data.detail||"Prediction failed");
  message.textContent=data.message||"";
  latestLandmarks=data.landmarks||{hand:[],body:[],face:[]};drawLandmarks();
  if(data.recognized){
   signResult.textContent=data.sign;translationResult.textContent=data.translation||"Translation unavailable";
   confidenceResult.textContent=`Confidence: ${(data.confidence*100).toFixed(1)}%`;
   setTranslationAudio(data.audio_url||"");
   await considerSaving(data);
  }else if(data.sign==="Unknown"){
   signResult.textContent="Unknown";confidenceResult.textContent=`Confidence: ${(data.confidence*100).toFixed(1)}%`;translationResult.textContent="—";
   setTranslationAudio("");
  }
 }catch(e){message.textContent="Recognition error: "+e.message}
 finally{busy=false}
}
async function considerSaving(data){
 if(candidate===data.sign)count++;else{candidate=data.sign;count=1;lastSaved="";lastAutoSpoken="";}
 if(count>=3&&lastAutoSpoken!==data.sign){
  lastAutoSpoken=data.sign;
  addSentenceWord(data);
  if(data.audio_url){
   setTranslationAudio(data.audio_url);
   playCurrentAudio();
  }else{
   message.textContent="Sign recognized and translated. No recorded Meiteilon audio is linked yet; use the replay button after adding a clip.";
  }
 }
 if(count>=3&&lastSaved!==data.sign){
  const response=await fetch("/api/history",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({sign:data.sign,confidence:data.confidence})});
  if(response.ok){lastSaved=data.sign;await loadHistory()}
 }
}
async function loadHistory(){
 try{
  const response=await fetch("/api/history"),rows=await response.json();historyBody.innerHTML="";
  if(!rows.length){historyBody.innerHTML='<tr><td colspan="4">No records yet.</td></tr>';return}
  for(const row of rows){const tr=document.createElement("tr");for(const value of [row.recognized_at,row.sign,row.translation,(row.confidence*100).toFixed(1)+"%"]){const td=document.createElement("td");td.textContent=value;tr.appendChild(td)}historyBody.appendChild(tr)}
 }catch(e){message.textContent="Could not load history: "+e.message}
}
loadHistory();
