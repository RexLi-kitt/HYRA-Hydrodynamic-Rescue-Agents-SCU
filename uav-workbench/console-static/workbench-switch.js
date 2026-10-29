'use strict';
const switchEight=document.getElementById('switchEight');
const switchLarge=document.getElementById('switchLarge');
const eightWorkspace=document.getElementById('eightWorkspace');
const largeWorkspace=document.getElementById('largeWorkspace');
const largeFrame=document.getElementById('largeFrame');
function selectWorkbench(large,updateHash=true){
  eightWorkspace.hidden=large;largeWorkspace.hidden=!large;
  switchEight.setAttribute('aria-pressed',String(!large));switchLarge.setAttribute('aria-pressed',String(large));
  if(large&&!largeFrame.src)largeFrame.src='/large-flood?embedded=1';
  if(updateHash)history.replaceState(null,'',large?'#large-flood':'#eight');
}
switchEight.addEventListener('click',()=>selectWorkbench(false));
switchLarge.addEventListener('click',()=>selectWorkbench(true));
window.addEventListener('hashchange',()=>selectWorkbench(location.hash==='#large-flood',false));
selectWorkbench(location.hash==='#large-flood',false);
