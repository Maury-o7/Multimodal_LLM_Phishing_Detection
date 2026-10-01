import json
import glob
import time
from tqdm import tqdm
import os
from MMLLM_Common import *
from google import genai
from google.genai import errors


class MMLLM_Gemini:
    def __init__(self, str_api_key: str):
        self.str_api_key = str_api_key
        self.str_phase1_system_msg: dict
        self.str_phase1_res_format: dict
        self.str_phase2_system_msg: dict

        # Updated: the old `google.generativeai` package (genai.configure /
        # genai.GenerativeModel) is deprecated. This now uses the current
        # unified SDK, `google-genai` (pip install google-genai).
        self.str_model = "gemini-2.5-flash"
        self.client = genai.Client(api_key=str_api_key)
        return

    def load_prompt_text(self, input_mode: InputMode):
        # Prompts
        str_phase1_prompt_path = dict_system_prompt_path.get(input_mode)
        assert str_phase1_prompt_path is not None, f"Unknown Input mode {input_mode}"

        with open(str_phase1_prompt_path, encoding='utf-8') as f_read:
            str_phase1_system_prompt = f_read.read()
            self.str_phase1_system_msg = str_phase1_system_prompt

        # Response
        str_phase1_response_prompt_path = dict_response_prompt_path.get(input_mode)
        assert str_phase1_response_prompt_path is not None, f"Unknown Input mode {input_mode}"

        with open(str_phase1_response_prompt_path, encoding='utf-8') as f_read:
            str_res_format = f_read.read()
            self.str_phase1_res_format = '\n\n' + str_res_format

        # Phase 2 Prompt
        # FIX: this path lives in dict_system_prompt_path, not
        # dict_response_prompt_path (which has no Phase2Mode.Phase2 key
        # and would always return None here, failing the assert below).
        str_phase2_system_prompt_path = dict_system_prompt_path.get(Phase2Mode.Phase2)
        assert str_phase2_system_prompt_path is not None, f"Unknown Input mode {Phase2Mode.Phase2}"

        with open(str_phase2_system_prompt_path, encoding='utf-8') as f_read:
            str_phase2_system_prompt = f_read.read()
            self.str_phase2_system_msg = str_phase2_system_prompt
        return

    def create_identification_prompt(self, input_mode: InputMode, encoded_image, html_content):
        '''
        First phase: Prompt for Brand identification. Gemini format.
        '''
        str_resource_msg = "Here are the provided resources: "
        list_phase1_system_msg = [self.str_phase1_system_msg, str_resource_msg]

        if input_mode == InputMode.SS:
            # Add Image
            list_phase1_system_msg.append(encoded_image)
        elif input_mode == InputMode.HTML:
            list_phase1_system_msg.append(html_content)
        elif input_mode == InputMode.BOTH:
            list_phase1_system_msg.append(html_content)
            list_phase1_system_msg.append(encoded_image)

        list_phase1_system_msg.append(self.str_phase1_res_format)

        return list_phase1_system_msg

    def create_brandcheck_prompt(self, str_groundtruth: str, str_prediction: str):
        '''
        Second phase: Prompt for Brand comparison and Phishing classification. Gemini format.
        '''
        str_phase2_data = f"Ground Truth: \"{str_groundtruth}\"\n\"Prediction:\"{str_prediction}\""
        list_phase2_system_msg = [self.str_phase2_system_msg, str_phase2_data]
        return list_phase2_system_msg

    def phase1_brand_identification(self, input_dataset: InputDataset):
        str_dataset = input_dataset
        list_data_dir = glob.glob(f'{str_input_dir_base}/{str_dataset}/*/*/')  # ../input/<dataset>/<brand>/<hash>/
        list_data_dir.sort()

        for str_data_dir in tqdm(list_data_dir, desc=f'{str_dataset}'):
            str_data_dir = str_data_dir.replace('\\', '/')
            list_prop = str_data_dir.split('/')
            str_ss_path = str_data_dir + '/screenshot_aft.png'
            str_html_path = str_data_dir + '/add_info.json'

            str_hash = list_prop[-2]
            str_brand = list_prop[-3]

            if not os.path.exists(str_ss_path) or not os.path.exists(str_html_path):
                # Error sample
                continue

            image = crop_encode_image_PIL(str_ss_path)

            # HTML Input
            with open(str_html_path) as f_read:
                str_html_info = json.load(f_read)['html_brand_info']

            for input_mode in InputMode:
                self.load_prompt_text(input_mode)  # Following input_mode

                str_output_dir = os.path.join(str_output_dir_base, str_dataset, 'Phase1_Gemini', input_mode, str_brand)
                if not os.path.exists(str_output_dir):
                    os.makedirs(str_output_dir)

                # Gemini
                list_model_prompt = self.create_identification_prompt(input_mode, image, str_html_info)

                try:
                    response = self.client.models.generate_content(
                        model=self.str_model,
                        contents=list_model_prompt
                    )
                except errors.ClientError as e:
                    if e.code == 429:
                        # Rate limit / quota exceeded
                        print(f'[Warning] {str_hash} rate limited (429)')
                        time.sleep(60 * 60)
                    else:
                        print(f'[Warning] {str_hash} ClientError: {e}')
                    continue
                except errors.ServerError as e:
                    print(f'[Warning] {str_hash} ServerError: {e}')
                    time.sleep(60 * 60)
                    continue
                except errors.APIError as e:
                    print(f'[Warning] {str_hash} APIError: {e}')
                    continue

                if getattr(response, 'prompt_feedback', None) and response.prompt_feedback.block_reason:
                    dict_res_data = format_model_response(str_hash, '', True, False)
                elif not response.candidates or not response.candidates[0].content.parts:
                    dict_res_data = format_model_response(str_hash, str(response.candidates), False, True)
                else:
                    # Normal
                    str_res: str
                    str_res = response.text

                    try:
                        dict_res_data = format_model_response(str_hash, str_res)
                    except Exception:
                        dict_res_data = format_model_response(str_hash, 'Safety Error', False, True)

                if response.usage_metadata:
                    dict_res_data['prompt_token_count'] = response.usage_metadata.prompt_token_count
                    dict_res_data['candidates_token_count'] = response.usage_metadata.candidates_token_count
                    dict_res_data['total_token_count'] = response.usage_metadata.total_token_count

                str_output_file_path = os.path.join(str_output_dir, f"{str_hash}.json")
                with open(str_output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(dict_res_data, f, indent=4)
        return

    def phase2_phishing_classification(self, input_dataset: InputDataset):
        str_dataset = input_dataset
        # No need to Load Phase2 Prompt. It is already loaded in the instance at Phase1.

        # Output: Summary
        str_output_summary_path = os.path.join(str_output_dir_base, str_dataset, 'Phase2_Gemini', "Phase2_Res_Summary.csv")
        if not os.path.exists(os.path.dirname(str_output_summary_path)):
            os.makedirs(os.path.dirname(str_output_summary_path))
        if not os.path.exists(str_output_summary_path):
            with open(str_output_summary_path, 'w') as f_summary:
                str_phase2_res_summary_hdr = f'Dataset,InputMode,Brand,Hash,Phase1Pred,Phase2Matched\n'
                f_summary.write(str_phase2_res_summary_hdr)

        for input_mode in InputMode:
            # FIX: str_brand used to be referenced here before it was ever
            # assigned. We now glob across all brands for this input_mode
            # instead of a single, not-yet-defined brand.
            str_input_dir = os.path.join(str_output_dir_base, str_dataset, 'Phase1_Gemini', input_mode)
            list_input_path = glob.glob(f'{str_input_dir}/*/*.json')  # ./output/<dataset>/<input_mode>/<brand>/<hash>.json
            list_input_path = [x.replace('\\', '/') for x in list_input_path]
            list_input_path.sort()

            for str_input_path in list_input_path:
                list_prop = str_input_path.split('/')

                str_hash = list_prop[-1].replace('.json', '')
                str_brand = list_prop[-2]

                str_output_dir = os.path.join(str_output_dir_base, str_dataset, 'Phase2_Gemini', input_mode, str_brand)
                if not os.path.exists(str_output_dir):
                    os.makedirs(str_output_dir)

                # Brand Prediction at Phase 1
                # FIX: the file was being opened and json.load()'d twice,
                # but the second call on an already-consumed file handle
                # would fail. Load it once and read both keys from it.
                try:
                    with open(str_input_path) as f_read:
                        dict_phase1_result = json.load(f_read)
                        str_phase1_pred = dict_phase1_result['Brand']
                        b_phase1_error = dict_phase1_result['Error']
                except Exception:
                    print(f'[Warning] Broken Phase 1 result: {str_input_path}')
                    continue

                if b_phase1_error == True:
                    continue

                # FIX: create_brandcheck_prompt only takes (groundtruth, prediction).
                list_model_prompt = self.create_brandcheck_prompt(str_brand, str_phase1_pred)

                try:
                    response = self.client.models.generate_content(
                        model=self.str_model,
                        contents=list_model_prompt
                    )
                except errors.ClientError as e:
                    if e.code == 429:
                        print(f'[Warning] {str_hash} rate limited (429)')
                        time.sleep(60 * 60)
                    else:
                        print(f'[Warning] {str_hash} ClientError: {e}')
                    continue
                except errors.ServerError as e:
                    print(f'[Warning] {str_hash} ServerError: {e}')
                    time.sleep(60 * 60)
                    continue
                except errors.APIError as e:
                    print(f'[Warning] {str_hash} APIError: {e}')
                    continue

                if getattr(response, 'prompt_feedback', None) and response.prompt_feedback.block_reason:
                    dict_res_data = format_phase2_response('', True, False)
                elif not response.candidates or not response.candidates[0].content.parts:
                    dict_res_data = format_phase2_response(str(response.candidates), False, True)
                else:
                    # Normal
                    str_res: str
                    str_res = response.text

                    try:
                        dict_res_data = format_phase2_response(str_res, False, False)
                    except Exception:
                        dict_res_data = format_phase2_response(str_res, False, True)

                if response.usage_metadata:
                    dict_res_data['prompt_token_count'] = response.usage_metadata.prompt_token_count
                    dict_res_data['candidates_token_count'] = response.usage_metadata.candidates_token_count
                    dict_res_data['total_token_count'] = response.usage_metadata.total_token_count

                str_output_file_path = os.path.join(str_output_dir, f"{str_hash}.json")
                with open(str_output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(dict_res_data, f, indent=4)

                str_phase2_res_summary = f"{str_dataset},{input_mode},{str_brand},{str_hash},{str_phase1_pred},{dict_res_data['BrandMatched']}\n"
                with open(str_output_summary_path, 'a') as f_summary:
                    f_summary.write(str_phase2_res_summary)
        return