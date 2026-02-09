#!/usr/bin/env python3
"""
Inference Script with Excel Output
Based on the working evaluation code from LLaVA
"""

import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from DentFound.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from DentFound.conversation import conv_templates, SeparatorStyle
from DentFound.model.builder import load_pretrained_model
from DentFound.utils import disable_torch_init
from DentFound.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import warnings
warnings.filterwarnings("ignore")


class DentFoundInference:
    def __init__(self, args):
        # Model initialization
        disable_torch_init()
        
        self.args = args
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)
        
        print(f"Loading model from {model_path}...")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, 
            args.model_base, 
            model_name
        )
        self.model = self.model.to("cuda")
        self.model = self.model.half()  # force everything to float16
        print(f"Model loaded successfully")
        print(f"Model config: mm_use_im_start_end={self.model.config.mm_use_im_start_end}")
        
    def process_single_sample(self, sample_data):
        """Process a single sample and generate prediction"""
        # Extract question and ground truth from conversations
        human_question = ""
        ground_truth = ""
        
        for conv in sample_data.get('conversations', []):
            if conv['from'] == 'human':
                human_question = conv['value'].replace('<image>', '').strip()
            elif conv['from'] == 'gpt':
                ground_truth = conv['value']
        
        # Get image path
        image_path = sample_data['image']
        
        # Check if image exists
        if not os.path.exists(image_path):
            return {
                'prediction': "",
                'status': "Image Not Found",
                'error': f"File not found: {image_path}"
            }
        
        # Prepare the question with image token
        qs = human_question
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        
        # Setup conversation
        conv = conv_templates[self.args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Tokenize
        input_ids = tokenizer_image_token(
            prompt, 
            self.tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        ).unsqueeze(0).cuda()
        
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]
        
        print(input_ids.shape, image_tensor.unsqueeze(0).half().cuda().shape, image.size)
        # Generate response
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if self.args.temperature > 0 else False,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                num_beams=self.args.num_beams,
                max_new_tokens=self.args.max_new_tokens,
                use_cache=True
            )
        
        # Decode output
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(outputs)
        # Clean output - remove the input prompt if it's included
        if outputs.startswith(human_question):
            outputs = outputs[len(human_question):].strip()
        
        # Remove any conversation separators
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        if stop_str and stop_str in outputs:
            outputs = outputs.split(stop_str)[0].strip()
        
        return {
            'prediction': outputs,
            'status': "Success",
            'error': ""
        }
            
    def process_test_json(self, json_path, output_excel, save_intermediate=True, batch_save=100, resume_from=0, pre_num = 0 ):
        """Process test JSON and save results to Excel"""
        
        # Load test data
        print(f"Loading test data from {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # Initialize or load existing results
        results = []
        if resume_from > 0:
            json_output = output_excel.replace('.xlsx', '_results.json')
            if os.path.exists(json_output):
                print(f"Loading existing results from {json_output}")
                with open(json_output, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                print(f"Loaded {len(results)} existing results, resuming from index {resume_from}")
        
        total = len(test_data)
        successful_count = 0
        failed_count = 0
        
        print(f"Processing {total} samples...")
        print(f"Will save intermediate results every {batch_save} samples")
        if pre_num > 0:
            total = pre_num
        # Process each sample
        for idx in tqdm(range(resume_from, total), desc="Processing samples"):
            item = test_data[idx]
            sample_id = item.get('id', f'sample_{idx+1}')
            
            print(f"\n[{idx+1}/{total}] Processing: {sample_id}")
            
            # Extract information from conversations
            human_question = ""
            ground_truth = ""
            
            for conv in item.get('conversations', []):
                if conv['from'] == 'human':
                    human_question = conv['value'].replace('<image>', '').strip()
                elif conv['from'] == 'gpt':
                    ground_truth = conv['value']
            
            # Process the sample
            result_dict = self.process_single_sample(item)
            
            # Update counters
            if result_dict['status'] == "Success":
                successful_count += 1
                print(f"  ✅ Success: Generated {len(result_dict['prediction'])} characters")
                if len(result_dict['prediction']) > 0:
                    preview = result_dict['prediction'][:100] + "..." if len(result_dict['prediction']) > 100 else result_dict['prediction']
                    print(f"  Preview: {preview}")
            else:
                failed_count += 1
                print(f"  ❌ {result_dict['status']}: {result_dict['error']}")
            
            # Create result entry
            result = {
                'ID': sample_id,
                'Image_Path': item['image'],
                'Image_Name': os.path.basename(item['image']),
                'Question': human_question,
                'Ground_Truth': ground_truth,
                'Model_Prediction': result_dict['prediction'],
                'Status': result_dict['status'],
                'Error_Message': result_dict['error'],
                'GT_Length': len(ground_truth),
                'Pred_Length': len(result_dict['prediction']),
                'Match': ground_truth.strip().lower() == result_dict['prediction'].strip().lower()
            }
            
            results.append(result)
            
            # Print running statistics
            total_processed = idx - resume_from + 1
            if total_processed > 0:
                success_rate = (successful_count / total_processed) * 100
                print(f"  📊 Running stats: Success rate = {success_rate:.1f}% ({successful_count}/{total_processed})")
            
            # Save intermediate results
            if (idx + 1) % batch_save == 0:
                print(f"\n💾 Saving intermediate results at sample {idx + 1}...")
                self._save_results(results, output_excel, save_intermediate)
                print(f"  Progress: {idx + 1}/{total} completed ({(idx + 1)/total*100:.1f}%)")
        
        # Final save
        print("\n💾 Saving final results...")
        df = self._save_results(results, output_excel, save_intermediate)
        
        return df
    
    def _save_results(self, results: List[Dict], output_excel: str, save_json: bool) -> pd.DataFrame:
        """Save results to Excel and optionally JSON"""
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Calculate statistics
        total_samples = len(df)
        successful = len(df[df['Status'] == 'Success']) if 'Status' in df.columns else 0
        failed = len(df[df['Status'] == 'Failed']) if 'Status' in df.columns else 0
        not_found = len(df[df['Status'] == 'Image Not Found']) if 'Status' in df.columns else 0
        
        # Calculate exact matches only for successful predictions
        if successful > 0:
            exact_matches = len(df[(df['Match'] == True) & (df['Status'] == 'Success')])
        else:
            exact_matches = 0
        
        # Create statistics
        stats = {
            'Metric': [
                'Total Samples',
                'Successful Inferences', 
                'Failed Inferences',
                'Images Not Found',
                'Exact Matches',
                'Exact Match Rate (%)',
                'Success Rate (%)',
                'Average GT Length',
                'Average Prediction Length',
                'Inference Date',
                'Temperature',
                'Top-p',
                'Max New Tokens',
                'Model Path'
            ],
            'Value': [
                total_samples,
                successful,
                failed,
                not_found,
                exact_matches,
                f"{(exact_matches/successful*100):.2f}" if successful > 0 else "0.00",
                f"{(successful/total_samples*100):.2f}" if total_samples > 0 else "0.00",
                f"{df['GT_Length'].mean():.2f}" if len(df) > 0 and 'GT_Length' in df.columns else "0.00",
                f"{df[df['Status'] == 'Success']['Pred_Length'].mean():.2f}" if successful > 0 else "0.00",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                self.args.temperature,
                self.args.top_p,
                self.args.max_new_tokens,
                self.args.model_path
            ]
        }
        stats_df = pd.DataFrame(stats)
        
        # Save to Excel
        try:
            with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Results', index=False)
                stats_df.to_excel(writer, sheet_name='Statistics', index=False)
            print(f"✅ Excel results saved to: {output_excel}")
        except Exception as e:
            print(f"⚠️  Error saving Excel: {str(e)}")
            csv_path = output_excel.replace('.xlsx', '.csv')
            df.to_csv(csv_path, index=False)
            print(f"✅ Results saved as CSV: {csv_path}")
        
        # Save JSON if requested
        if save_json:
            json_output = output_excel.replace('.xlsx', '_results.json')
            with open(json_output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✅ JSON results saved to: {json_output}")
        
        # Print summary
        print("\n" + "="*60)
        print("📊 INFERENCE SUMMARY")
        print("="*60)
        print(f"Total Samples: {total_samples}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️  Not Found: {not_found}")
        if successful > 0:
            print(f"🎯 Exact Matches: {exact_matches}/{successful} ({(exact_matches/successful*100):.2f}%)")
            print(f"📈 Success Rate: {(successful/total_samples*100):.2f}%")
        print("="*60)
        
        return df


def main():
    parser = argparse.ArgumentParser(description='LLaVA Inference with Excel Output')
    
    # Model arguments
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the fine-tuned model checkpoint")
    parser.add_argument("--model-base", type=str, default=None,
                        help="Base model path (if using LoRA)")
    
    # Data arguments
    parser.add_argument("--test-json", type=str, required=True,
                        help="Path to test JSON file")
    parser.add_argument("--output-excel", type=str, default="inference_results.xlsx",
                        help="Path to save Excel results")
    
    # Generation arguments
    parser.add_argument("--conv-mode", type=str, default="llava_v1",
                        help="Conversation mode")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.7,
                        help="Top-p sampling parameter")
    parser.add_argument("--num_beams", type=int, default=1,
                        help="Number of beams for beam search")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    
    # Processing arguments
    parser.add_argument("--save-json", action="store_true",
                        help="Also save intermediate JSON results")
    parser.add_argument("--batch-save", type=int, default=100,
                        help="Save intermediate results every N samples")
    parser.add_argument("--resume-from", type=int, default=0,
                        help="Resume from sample index (0-based)")
    parser.add_argument("--pre_num", type=int, default=0,
                        help="prediactin from sample index (0-based)")                        
    
    args = parser.parse_args()
    
    print("🚀 Starting LLaVA Inference")
    print(f"📁 Model: {args.model_path}")
    print(f"📄 Test file: {args.test_json}")
    print(f"💾 Output: {args.output_excel}\n")
    
    # Initialize inference class
    inference = DentFoundInference(args)
    
    # Process test JSON and save to Excel
    df = inference.process_test_json(
        json_path=args.test_json,
        output_excel=args.output_excel,
        save_intermediate=args.save_json,
        batch_save=args.batch_save,
        resume_from=args.resume_from,
        pre_num = args.pre_num,
    )
    
    print(f"\n✅ Final results saved to: {args.output_excel}")
    print("🎉 Inference completed successfully!")


if __name__ == "__main__":
    main()
