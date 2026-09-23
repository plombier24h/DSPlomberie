/* Moteur de transcription Whisper, executé dans un Web Worker.
   Le modèle tourne dans le navigateur : aucun audio ne quitte l'appareil. */
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

env.allowLocalModels = false;
env.useBrowserCache = true;          // le modèle n'est téléchargé qu'une fois par appareil

let transcriber = null;
let modeleCourant = '';

self.onmessage = async (e) => {
  const d = e.data || {};

  if(d.type === 'init'){
    const modele = d.model || 'onnx-community/whisper-base';
    if(transcriber && modeleCourant === modele){ self.postMessage({ type:'ready' }); return; }
    try{
      let device = 'wasm';
      try{ if(navigator.gpu && await navigator.gpu.requestAdapter()) device = 'webgpu'; }catch(_){}
      transcriber = await pipeline('automatic-speech-recognition', modele, {
        dtype: device === 'webgpu' ? 'fp32' : 'q8',
        device,
        progress_callback: (p) => self.postMessage({ type:'progress', p })
      });
      modeleCourant = modele;
      self.postMessage({ type:'ready', device });
    }catch(err){
      self.postMessage({ type:'error', message: String(err && err.message || err) });
    }
    return;
  }

  if(d.type === 'audio'){
    if(!transcriber) return;
    try{
      const out = await transcriber(d.audio, {
        language: 'french',
        task: 'transcribe',
        return_timestamps: false,
        chunk_length_s: 30
      });
      self.postMessage({ type:'text', text: String((out && out.text) || '').trim() });
    }catch(err){
      self.postMessage({ type:'error', message: String(err && err.message || err) });
    }
  }
};
