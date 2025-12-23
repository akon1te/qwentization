import re
import time
import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, pipeline

from datetime import datetime
from typing import Any, Dict, cast


class MMLUEvaluator:

    def __init__(self, model, tokenizer, device="cuda",
                 split="dev", per_subject_samples=10, seed=42,
                 model_name=None, experiment_name=None):

        self.split_name = split
        self.per_subject_samples = per_subject_samples
        self.seed = seed
        self.device = device

        self.model = model
        self.tokenizer = tokenizer
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.pipe = None
        
        
        ds = load_dataset("cais/mmlu", "all", split=self.split_name)
        print(f"  Загружена {self.split_name} выборка")

        self.dataset: Dataset = cast(Dataset, ds)
        

        self.model_name = model_name or getattr(model.config, 'model_type', 'unknown_model')
        self.experiment_name = experiment_name or f"{self.model_name}_dev_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.total_questions = len(self.dataset)
    
        subj_set = set()
        for ex in self.dataset:
            try:
                ex_d = cast(Dict[str, Any], ex)
                subj_val = ex_d.get("subject", None)
                if isinstance(subj_val, str):
                    subj_set.add(subj_val)
            except Exception:
                pass
        self.subjects = sorted(subj_set)
        self.questions_per_subject = {}
        
        for subject in self.subjects:
            subject_data = cast(Dataset, self.dataset.filter(lambda x: x.get("subject", None) == subject))
            self.questions_per_subject[subject] = len(subject_data)
        
        print(f"  Всего вопросов в {self.split_name} выборке: {self.total_questions}")
        print(f"  Количество предметов: {len(self.subjects)}")
        
        
        self.zero_shot_template = """The following are multiple choice questions (with answers) about {subject}.
 
 Question: {question}
 A. {choice_a}
 B. {choice_b}
 C. {choice_c}
 D. {choice_d}
 Please choose the best answer and reply with a single letter only (A, B, C, or D).
 Answer:"""
        

        self.prompt_template = self.zero_shot_template
        
        print(f"Инициализация завершена. Эксперимент: {self.experiment_name}\n")

    def set_prompt_style(self, style="zero_shot"):
        self.prompt_template = self.zero_shot_template


    def extract_answer(self, generated_text):

        if "</think>" in generated_text:
            generated_text = generated_text.split("</think>")[-1].strip()
        try:
            generated_text = re.sub(r"<think>.*?(</think>|$)", "", generated_text, flags=re.DOTALL).strip()
        except Exception:
            pass
        
        if "Answer:" in generated_text:
            parts = generated_text.split("Answer:")
            if len(parts) > 1:
                generated_text = parts[-1].strip()
        
        patterns = [
            r'answer[\s:\-]*([ABCD])',
            r'([ABCD])\)',            
            r'\s([ABCD])[\s\.\n]',    
            r'^([ABCD])\b',           
            r'([ABCD])(?=[\s\.\n]|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, generated_text, re.IGNORECASE)
            if match:
                extracted = match.group(1).upper()

                if extracted in ['A', 'B', 'C', 'D']:
                    return extracted
        
        return ""
        
    def _predict_by_logits(self, text: str) -> str:

        try:
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1]
        
            candidates = []
            for lbl in ["A", "B", "C", "D"]:
                toks_space = self.tokenizer.encode(f" {lbl}", add_special_tokens=False)
                toks_plain = self.tokenizer.encode(lbl, add_special_tokens=False)
                if len(toks_space) == 1:
                    candidates.append((lbl, toks_space[0]))
                if len(toks_plain) == 1:
                    candidates.append((lbl, toks_plain[0]))
        
            if not candidates:
                return ""
        
            best_lbl = ""
            best_logit = -1e30
            for lbl, tid in candidates:
                val = logits[tid].item()
                if val > best_logit:
                    best_logit = val
                    best_lbl = lbl
            return best_lbl
        except Exception:
            return ""
        

    def _normalize_label(self, ans):

        if isinstance(ans, int):
            if 0 <= ans <= 3:
                return "ABCD"[ans]
        if isinstance(ans, str):
            s = ans.strip().upper()
            if s in ["A", "B", "C", "D"]:
                return s
            if s in ["0", "1", "2", "3"]:
                return "ABCD"[int(s)]
        return ""


    def evaluate_subject(self, subject_name, subject_data, step=0,
                        subject_index=None, total_subjects=None):

        correct = 0
        total = 0

        subj_time_ms_total = 0.0
        subj_tokens_total = 0
        subj_seq_count = 0
        
        # Reset peak memory stats for this subject
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        for item in subject_data:
            prompt = self.prompt_template.format(
                subject=item["subject"],
                question=item["question"],
                choice_a=item["choices"][0],
                choice_b=item["choices"][1],
                choice_c=item["choices"][2],
                choice_d=item["choices"][3]
            )
            
            try:
                text = prompt
                
                t0 = time.perf_counter()
                predicted_answer = self._predict_by_logits(text)
                generated_text = ""
                new_tokens = 0
                
                if not predicted_answer:
                    inputs = self.tokenizer(text, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        gen_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=8,
                            do_sample=False,
                            temperature=0.0,
                            use_cache=False, 
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                    
                    new_tokens = int(gen_ids.shape[1] - inputs["input_ids"].shape[1])
                    generated_text = self.tokenizer.decode(
                        gen_ids[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True
                    ).strip()
                    predicted_answer = self.extract_answer(generated_text)
                
                t1 = time.perf_counter()
                subj_time_ms_total += (t1 - t0) * 1000.0
                subj_tokens_total += new_tokens
                subj_seq_count += 1
                
                correct_answer_raw = item["answer"]
                correct_answer = self._normalize_label(correct_answer_raw)
                
                if predicted_answer == correct_answer:
                    correct += 1
                total += 1
                
                
            except Exception as e:
                print(f"\nОшибка при обработке вопроса: {e}")
                continue
        
        accuracy = correct / total if total > 0 else 0.0
        avg_tokens_per_seq = (subj_tokens_total / total) if total > 0 else 0.0
        avg_latency_ms_per_seq = (subj_time_ms_total / total) if total > 0 else 0.0
        
        peak_memory_mb = 0.0
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated(device='cuda:0') / (1024 * 1024)
         
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "avg_tokens_per_seq": avg_tokens_per_seq,
            "avg_latency_ms_per_seq": avg_latency_ms_per_seq,
            "peak_memory_mb": peak_memory_mb
        }


    def evaluate(self, step=0, use_few_shot=False):
        """
        Основной метод оценки модели на dev выборке MMLU.

        Args:
            step (int): Шаг для логирования в TensorBoard.
            use_few_shot (bool): Использовать few-shot промпт.

        Returns:
            dict: Итоговые результаты оценки.
        """

        print(f"Эксперимент: {self.experiment_name}")
        print(f"Модель: {self.model_name}")
        print(f"Всего вопросов в {self.split_name}: {self.total_questions}")
        print(f"Количество предметов: {len(self.subjects)}")
        print(f"Промпт стиль: {'few-shot (2 примера)' if use_few_shot else 'zero-shot'}")
        
        # Устанавливаем стиль промпта
        self.set_prompt_style("few_shot" if use_few_shot else "zero_shot")
        
        results = {}
        all_correct = 0
        all_total = 0
        all_tokens_total = 0.0
        all_time_ms_total = 0.0
         
        max_peak_memory_mb = 0.0
         
        # Оценка по каждому предмету в выбранном split с подсэмплингом per_subject_samples
        for idx, subject in tqdm(enumerate(self.subjects)):
            subject_data_full = cast(Dataset, self.dataset.filter(lambda x: x.get("subject", None) == subject))
            subject_data = subject_data_full
            if isinstance(self.per_subject_samples, int) and self.per_subject_samples > 0:
                k = min(len(subject_data_full), self.per_subject_samples)
                subject_data = subject_data_full.shuffle(seed=self.seed).select(range(k))
            
            subject_result = self.evaluate_subject(
                subject, subject_data, step, idx, len(self.subjects)
            )
            
            results[subject] = subject_result
            all_correct += subject_result["correct"]
            all_total += subject_result["total"]
            all_tokens_total += subject_result.get("avg_tokens_per_seq", 0.0) * subject_result["total"]
            all_time_ms_total += subject_result.get("avg_latency_ms_per_seq", 0.0) * subject_result["total"]
            
            if subject_result.get("peak_memory_mb", 0.0) > max_peak_memory_mb:
                max_peak_memory_mb = subject_result.get("peak_memory_mb", 0.0)

        
        overall_accuracy = all_correct / all_total if all_total > 0 else 0.0
        overall_avg_tokens_per_seq = (all_tokens_total / all_total) if all_total > 0 else 0.0
        overall_avg_latency_ms_per_seq = (all_time_ms_total / all_total) if all_total > 0 else 0.0
        qps = (all_total / (all_time_ms_total / 1000.0)) if all_time_ms_total > 0 else 0.0
       
        results["overall"] = {
            "accuracy": overall_accuracy,
            "correct": all_correct,
            "total": all_total,
            "avg_tokens_per_seq": overall_avg_tokens_per_seq,
            "avg_latency_ms_per_seq": overall_avg_latency_ms_per_seq,
            "questions_per_second": qps,
            "peak_memory_mb": max_peak_memory_mb
        }
        
        subject_accuracies = []
        for subject, result in results.items():
            if subject != "overall":
                subject_accuracies.append(result['accuracy'])
        
        
        print(f"ОБЩАЯ ТОЧНОСТЬ: {overall_accuracy:.4f} ({overall_accuracy:.2%})")
        print(f"Правильных ответов: {all_correct} из {all_total}")
        print(f"Оценено предметов: {len([s for s in results.keys() if s != 'overall'])}")
        print(f"Пиковое потребление VRAM: {max_peak_memory_mb:.2f} MB")
        
        return results
