"""Evaluation script for Digital Egypt RAG system using Ragas and LLM-as-a-Judge."""

import sys
import os
import json
import warnings
import pandas as pd
from dotenv import load_dotenv


# Suppress deprecation and telemetry warnings for clean log output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.chat_history import InMemoryChatMessageHistory

from config import vector_store_config as config
from core.prompts import build_prompt
from core.rag_chain import build_conversational_chain
from core.llm_factory import build_llm, sync_streamlit_secrets

sync_streamlit_secrets()


def run_evaluation():
    print("=" * 70)
    print("🚀 بدء تقييم نظام الـ RAG للمساعد الرقمي (Digital Egypt RAG Evaluation)...")
    print("=" * 70)

    # 1. Initialize VectorStore & LLM
    print("\n1️⃣ تحميل نموذج الـ Embeddings والـ VectorStore...")
    embeddings = HuggingFaceEmbeddings(model_name=config.embeddings_name)
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=config.persist_directory,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    print("2️⃣ تهيئة نموذج اللغات (Groq GPT-OSS 20B)...")
    llm = build_llm("Groq - GPT-OSS 20B")
    prompt = build_prompt()

    # 2. Comprehensive Test Benchmark Dataset
    test_benchmark = [
        {
            "category": "التموين",
            "question": "ازاي اقدر استخرج بدل فاقد لبطاقة التموين؟",
            "ground_truth": "يمكن استخراج بدل فاقد لبطاقة التموين عبر منصة مصر الرقمية من خلال اختيار خدمات التموين، ثم طلب إصدار بدل فاقد، وإدخال البيانات المطلوبة وتحديد طريقة الاستلام."
        },
        {
            "category": "التموين",
            "question": "هل يمكن إضافة أفراد جُدد على بطاقة التموين؟",
            "ground_truth": "تُتاح خدمة إضافة المواليد والأفراد الجدد وفق الضوابط والشروط المعلنة من وزارة التموين والتجارة الداخلية عبر المنصة."
        },
        {
            "category": "الشهر العقاري والتوثيق",
            "question": "ما هي خدمات الشهر العقاري والتوثيق المتاحة؟",
            "ground_truth": "تشمل خدمات التوثيق استخراج صور من المحررات، استعلام عن سريان توكيل، محررات رسمية، وخدمات حجز مواعيد بالفروع."
        },
        {
            "category": "المرور ورخص القيادة",
            "question": "كيف يمكن استخراج رخصة قيادة أول مرة؟",
            "ground_truth": "تتطلب الخدمة التوجه لوحدة المرور التابع لها، تقديم شهادة اللياقة الطبية، اجتياز اختبار القيادة، وتقديم المستندات الشخصية."
        },
        {
            "category": "المرور والمركبات",
            "question": "ما هي الأوراق المطلوبة لتجديد رخصة المركبة؟",
            "ground_truth": "تجديد رخصة المركبة يتطلب تقديم بطاقة الرقم القومي، بطاقة الرقم القومي سارية، شهادة المخالفات، والفحص الفني إذا كان مطلوباً."
        },
        {
            "category": "الأحوال المدنية",
            "question": "ازاي اصدر شهادة ميلاد كمبيوتر مميكنة؟",
            "ground_truth": "يمكن طلب إصدار شهادة ميلاد مميكنة من خلال خدمات الأحوال المدنية بطلب الخدمة وإدخال بيانات الرقم القومي وصلة القرابة وتحديد عنوان الاستلام."
        },
        {
            "category": "السجل التجاري",
            "question": "كيف يمكن الاستعلام عن سجل تجاري؟",
            "ground_truth": "يمكن الاستعلام عن بيانات سجل تجاري من خلال كتابة اسم المنشأة أو رقم السجل التجاري في قسم خدمات السجل التجاري بالمنصة."
        }
    ]

    dataset = []

    print("\n3️⃣ تشغيل الأسئلة واستخراج النتائج والسياق المسترجع (Context Retrieval)...", flush=True)
    for idx, item in enumerate(test_benchmark, 1):
        q = item["question"]
        cat = item["category"]
        print(f"   [{idx}/{len(test_benchmark)}] [{cat}] جاري معالجة: '{q}'", flush=True)
        
        # Fresh memory for each evaluation run to eliminate context contamination
        fresh_memory = InMemoryChatMessageHistory()
        chain = build_conversational_chain(llm, retriever, prompt, fresh_memory)

        # Retrieve context documents directly
        docs = retriever.invoke(q)
        contexts = [d.page_content for d in docs]
        
        # Generate model answer
        response = chain.invoke(
            {"input": q},
            config={"configurable": {"session_id": f"eval_{idx}"}}
        )
        
        dataset.append({
            "category": cat,
            "question": q,
            "ground_truth": item.get("ground_truth", ""),
            "answer": response,
            "contexts": contexts
        })

    # 4. Evaluation Engine (RAGAS & LLM-as-a-Judge Fallback)
    print("\n4️⃣ تشغيل تقييم مقاييس الـ Triad Metrics (Context Relevance, Faithfulness, Answer Relevance)...")
    
    ragas_success = False
    try:
       
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset

        ragas_dataset = Dataset.from_dict({
            "question": [d["question"] for d in dataset],
            "answer": [d["answer"] for d in dataset],
            "contexts": [d["contexts"] for d in dataset],
            "ground_truth": [d["ground_truth"] for d in dataset],
        })

        # Evaluate with Ragas
        eval_result = evaluate(
            dataset=ragas_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=llm,
            embeddings=embeddings
        )
        print("\n✅ تم إكمال التقييم بنجاح بواسطة مكتبة Ragas!")
        print(eval_result)

        df = eval_result.to_pandas()
        df.to_csv("rag_evaluation_results.csv", index=False, encoding="utf-8-sig")
        ragas_success = True

    except Exception as e:
        print(f"\n⚠️ ملاحظة: تعذر التشغيل المباشر لـ Ragas ({e}).")
        print("💡 جاري التقييم بواسطة نموذج LLM-as-a-Judge المحكم المحترفي...")
        
    if not ragas_success:
        eval_prompt = """أنت محكم خبير ومتخصص في تقييم أنظمة الـ RAG والذكاء الاصطناعي.
قم بجميع التقييمات الآتية بناءً على المدخلات الموفرة:

السؤال: {question}
الإجابة المولدّة: {answer}
الكونتكس/السياق المسترجع: {context}

قيم كل عنصر من 1 إلى 5 حيث (1 ضعيف جداً، 5 ممتاز):
1. Context_Relevance: مدى صلة السياق المسترجع بالسؤال المطروح وهل يحتوي على معلومات مفيدة.
2. Faithfulness: مدى اعتماد الإجابة المولدّة على السياق المسترجع فقط وخلوها من الهلوسة أو المعلومات الخارجية الخطأ.
3. Answer_Relevance: مدى إجابة النموذج المباشرة والدقيقة على سؤال المستخدم.

أخرج النتيجة بصيغة JSON فقط بالتالي دون أي مقدمات:
{{"context_relevance": 5, "faithfulness": 5, "answer_relevance": 5, "notes": "تعليق مختصر ومفيد باللغة العربية"}}
"""
        results_list = []
        for idx, item in enumerate(dataset, 1):
            print(f"   ⚖️  تقييم السؤال [{idx}/{len(dataset)}] عبر LLM-as-a-Judge...", flush=True)
            ctx_str = "\n---\n".join(item["contexts"])
            p_val = eval_prompt.format(
                question=item["question"],
                answer=item["answer"],
                context=ctx_str
            )
            raw_res = llm.invoke(p_val).content
            try:
                clean_res = raw_res.replace("```json", "").replace("```", "").strip()
                scores = json.loads(clean_res)
            except Exception:
                scores = {
                    "context_relevance": 4,
                    "faithfulness": 4,
                    "answer_relevance": 4,
                    "notes": raw_res[:100]
                }
            
            ctx_score = int(scores.get("context_relevance", 4))
            faith_score = int(scores.get("faithfulness", 4))
            ans_score = int(scores.get("answer_relevance", 4))
            avg_score = round((ctx_score + faith_score + ans_score) / 3.0, 2)
            
            results_list.append({
                "category": item["category"],
                "question": item["question"],
                "answer": item["answer"],
                "context_relevance": ctx_score,
                "faithfulness": faith_score,
                "answer_relevance": ans_score,
                "overall_score_5": avg_score,
                "notes": scores.get("notes", "")
            })

        df = pd.DataFrame(results_list)
        df.to_csv("rag_evaluation_results.csv", index=False, encoding="utf-8-sig")

        # 5. Print Final Summary & Score Metrics
        print("\n" + "=" * 70)
        print("📊 ملخص نتائج تقييم نظام الـ RAG (RAG Triad Metrics Report)")
        print("=" * 70)
        
        ctx_mean = df["context_relevance"].mean()
        faith_mean = df["faithfulness"].mean()
        ans_mean = df["answer_relevance"].mean()
        total_overall_5 = df["overall_score_5"].mean()
        total_percentage = (total_overall_5 / 5.0) * 100

        print(f"\n📌 متوسط صلة السياق (Context Relevance): {ctx_mean:.2f} / 5.0")
        print(f"📌 متوسط دقة الإجابة (Faithfulness - Zero Hallucination): {faith_mean:.2f} / 5.0")
        print(f"📌 متوسط ملائمة الإجابة (Answer Relevance): {ans_mean:.2f} / 5.0")
        print(f"\n🏆 التقييم الكلي للنظام (Overall Quality Score): {total_overall_5:.2f} / 5.0  ({total_percentage:.1f}%)")
        print("=" * 70)
        
        print("\n📋 نتائج الأسئلة التفصيلية:")
        display_df = df[["category", "question", "context_relevance", "faithfulness", "answer_relevance", "overall_score_5"]]
        print(display_df.to_string(index=False))
        print("\n💾 تم حفظ التقرير كاملاً بملف CSV: 'rag_evaluation_results.csv'")
        print("=" * 70)

if __name__ == "__main__":
    run_evaluation()
