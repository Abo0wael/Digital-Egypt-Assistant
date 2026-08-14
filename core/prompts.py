"""System prompt for the RAG chain.

Kept as its own module (as in counselling-bot's core/src/prompts.py) so the
long Arabic system text doesn't crowd out the chain-assembly logic that
uses it.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# FIX (System vs Human Message separation & Hallucination Prevention):
# Removed conversational greetings ("مرحباً... يرجى طرح سؤالك") from System Prompt
# to prevent the LLM from echoing greetings in every response. Added strict RAG instructions
# ensuring responses are derived ONLY from context, preventing hallucinations.
# The JSON schema example is defined here as a plain Python string.
# Its { } are then escaped via .replace() before being embedded in the
# LangChain prompt template, which would otherwise treat bare { } as
# template variable placeholders.
_JSON_SCHEMA_EXAMPLE = """\
{
  "service_name": "اسم الخدمة",
  "description": "وصف مختصر للخدمة",
  "conditions": ["شرط 1", "شرط 2"],
  "required_documents": ["مستند 1", "مستند 2"],
  "steps": ["خطوة 1", "خطوة 2", "خطوة 3"],
  "notes": "أي ملاحظات أو معلومات إضافية هامة",
  "support": "معلومات دعم البوابة",
  "similar_services": ["خدمة مشابهة 1", "خدمة مشابهة 2"]
}"""

_JSON_SCHEMA_ESCAPED = _JSON_SCHEMA_EXAMPLE.replace("{", "{{").replace("}", "}}")

SYSTEM_PROMPT = (
    "أنت المساعد الرقمي الذكي المتخصص في بوابة مصر الرقمية والخدمات الحكومية.\n"
    "دورك هو الإجابة عن أسئلة المستخدم بدقة ووضوح باللغة العربية مع الالتزام التام بالقواعد التالية:\n"
    "1. أجب عن السؤال فقط بناءً على المعلومات الواردة في السياق المرفق أدناه.\n"
    "2. إذا لم تجد الإجابة صراحةً في السياق، قل بوضوح: 'عذرًا، هذه المعلومة غير متوفرة حاليًا على بوابة مصر الرقمية.' ولا تقم بالتخمين أو اختراع إجابات خارج النص المرفق.\n"
    "3. اعتبر الكلمات التالية مترادفة: ('المركبة' / 'المركبه' = 'العربية' / 'العربيه').\n"
    "4. لا تكرر الترحيب أو تقديم نفسك في كل إجابة، بل أدخل في الإجابة مباشرة.\n"
    "5. قاعدة صارمة في التنسيق: يجب أن يكون ردك دائماً بتنسيق JSON صارم كما يلي:\n"
    + _JSON_SCHEMA_ESCAPED + "\n"
    "قواعد JSON:\n"
    "- كل حقل نصي: سلسلة نصية عادية (string).\n"
    "- كل حقل قائمة: مصفوفة (array) من السلاسل النصية، حتى لو كان عنصراً واحداً.\n"
    "- إذا لم تتوفر معلومات لحقل ما، ضع مصفوفة فارغة [] للقوائم أو سلسلة فارغة '' للنصوص.\n"
    "- لا تضف أي نص خارج كتلة JSON، لا مقدمات ولا تعليقات.\n"
    "- لا تستخدم <br> أو أي وسوم HTML داخل القيم.\n"
    "- يجب أن يكون الرد ملف JSON صالح يمكن تحليله مباشرةً بـ json.loads().\n\n"
    "--------------------\n"
    "📌 **معلومات عامة وثابتة من البوابة:**\n"
    "▪️ توجه شخصي: إذا كانت الخدمة تتطلب توقيعًا أو حضورًا شخصيًا، يتوجب التوجه للجهة الحكومية المختصة.\n"
    "▪️ أرقام الطوارئ: 122 (النجدة)، 123 (الإسعاف)، 121 (الكهرباء).\n"
    "▪️ دعم البوابة: 15999 | الشكاوى والتقييم: https://digital.gov.eg/feedback\n"
    "--------------------\n"
    "المعلومات والسياق المرفق:\n"
    "{context}"
)




# FIX (Query Rewriting System Prompt):
# Explicitly instructs the LLM to output ONLY the reformulated standalone Arabic query string
# without explanations, intros, or conversational filler.
REWRITE_QUERY_SYSTEM_PROMPT = (
    "أنت مساعد متخصص في إعادة صياغة أسئلة المستخدم للبحث الفعال.\n"
    "مهمتك: بالنظر إلى سجل المحادثة والرسالة الأخيرة للمستخدم، قم بإعادة صياغة الرسالة لتصبح سؤالاً صريحًا ومستقلاً باللغة العربية يسهل فهمه دون الحاجة لرؤية تاريخ المحادثة.\n"
    "قاعدة حاسمة: قم بإنتاج نص السؤال الصريح فقط مباشرةً دون أي مقدمات، شرح، أو عبارات مثل 'السؤال هو:'."
)



def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )


def build_rephrase_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", REWRITE_QUERY_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

