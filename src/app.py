import gradio as gr
import threading
import sys
import os

# Ensure src is in path if running from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT_MODEL_PATH, DEVICE
from model import VibeVoiceDemo
from locales import TRANS_TC

def run_app():
    print(f"Initializing VibeVoice on {DEVICE}...")
    # Initialize Model
    # Note: User can change model_path in config.py if needed
    demo = VibeVoiceDemo(model_path=DEFAULT_MODEL_PATH, device=DEVICE)

    with gr.Blocks(title=TRANS_TC["title_main"], theme=gr.themes.Soft()) as app:
        gr.Markdown(TRANS_TC["title_desc"])
        gr.Markdown(f"Running on: {DEVICE} | Model: {DEFAULT_MODEL_PATH}")
        
        with gr.Tabs():
            with gr.Tab(TRANS_TC["tab_gen"]):
                with gr.Row():
                    with gr.Column(scale=1):
                        num_speakers = gr.Slider(1, 4, value=2, step=1, label=TRANS_TC["label_num_speakers"])
                        
                        # Speakers Config Row
                        speakers_cfg = []
                        default_names = list(demo.available_voices.keys())[:4]
                        
                        # Create 4 slots
                        for i in range(4):
                            with gr.Group(visible=(i<2)) as grp:
                                gr.Markdown(f"**{TRANS_TC['label_speaker'].format(i+1)}**")
                                with gr.Row():
                                    # Preset Dropdown
                                    dd = gr.Dropdown(choices=list(demo.available_voices.keys()), 
                                                  value=default_names[i] if i < len(default_names) else None,
                                                  label=TRANS_TC["label_preset"], scale=2)
                                    # Custom Upload
                                    up = gr.Audio(type="filepath", label=TRANS_TC["label_upload"], scale=3)
                                speakers_cfg.append({'group': grp, 'dropdown': dd, 'upload': up})
                        
                        def update_vis(n):
                            return [gr.update(visible=(i<n)) for i in range(4)]
                            
                        # Extract group components for visibility updates
                        speaker_groups = [cfg['group'] for cfg in speakers_cfg]
                        num_speakers.change(update_vis, num_speakers, speaker_groups)
                        
                        # Advanced Settings Accordion
                        with gr.Accordion(TRANS_TC["group_advanced"], open=False):
                            # Speed Control
                            speed_slider = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label=TRANS_TC["label_speed"])
                        
                        script = gr.Textbox(label=TRANS_TC["label_script"], lines=5, 
                                          value="Speaker 1: Hello world!\nSpeaker 2: This is a test.",
                                          placeholder=TRANS_TC["placeholder_script"])
                        
                        with gr.Row():
                            btn = gr.Button(TRANS_TC["btn_generate"], variant="primary", scale=2)
                            stop_btn = gr.Button(TRANS_TC["btn_stop"], variant="stop", scale=1)
                        
                    with gr.Column(scale=1):
                        # SEPARATE OUTPUTS
                        gr.Markdown(f"### {TRANS_TC['label_status']}")
                        status = gr.Textbox(label=TRANS_TC["label_status"], show_label=False)
                        
                        output_stream = gr.Audio(label=TRANS_TC["label_stream"], autoplay=True, streaming=True)
                        output_file = gr.Audio(label=TRANS_TC["label_download"], type="filepath")
            
            with gr.Tab(TRANS_TC["tab_lib"]):
                gr.Markdown("### " + TRANS_TC["tab_lib"])
                gr.Markdown(TRANS_TC["tab_lib"]) # Simplified desc for now
                
                with gr.Group():
                    new_voice_name = gr.Textbox(label=TRANS_TC["label_voice_name"])
                    new_voice_file = gr.Audio(type="filepath", label=TRANS_TC["label_ref_audio"])
                    add_voice_btn = gr.Button(TRANS_TC["btn_save_voice"], variant="primary")
                
                lib_status = gr.Textbox(label=TRANS_TC["label_lib_status"])
                
                # Components to update
                dropdowns_to_update = [cfg['dropdown'] for cfg in speakers_cfg]
                
                add_voice_btn.click(
                    fn=demo.register_custom_voice,
                    inputs=[new_voice_name, new_voice_file],
                    outputs=[lib_status] + dropdowns_to_update
                )

        def run_gen(n_spk, s1_pre, s1_up, s2_pre, s2_up, s3_pre, s3_up, s4_pre, s4_up, spd, txt):
            # Collect lists
            speakers_list = [s1_pre, s2_pre, s3_pre, s4_pre]
            custom_uploads = [s1_up, s2_up, s3_up, s4_up]
            
            # Clear previous
            yield None, None, TRANS_TC["msg_starting"]
            
            yield from demo.generate_podcast_streaming(
                num_speakers=n_spk, script=txt, speakers_list=speakers_list, 
                cfg_scale=1.3, inference_steps=10, seed=42, disable_voice_cloning=False,
                custom_voice_paths=custom_uploads, speed_factor=spd
            )

        # Inputs: [num, (s1_dd, s1_up), (s2_dd, s2_up)... speed, script]
        inputs_list = []
        inputs_list.append(num_speakers)
        for cfg in speakers_cfg:
            inputs_list.append(cfg['dropdown'])
            inputs_list.append(cfg['upload'])
        inputs_list.append(speed_slider)
        inputs_list.append(script)
        
        btn.click(run_gen, 
                  inputs=inputs_list,
                  outputs=[output_stream, output_file, status])
        
        stop_btn.click(fn=demo.stop_audio_generation, outputs=[])
    
    # Launch
    app.launch(server_name='0.0.0.0',server_port=7860, share=False)

if __name__ == "__main__":
    run_app()
