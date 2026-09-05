"""Controlled academic vocabulary; no fuzzy course matching."""
MAJORS = {
"AI": ("ai", "artificial intelligence", "ذكاء اصطناعي", "الذكاء الاصطناعي", "ذكاء"),
"CS": ("cs", "computer science", "علوم الحاسب", "علوم حاسب", "حاسب", "حاسبات"),
"IS": ("is", "information systems", "information system", "نظم المعلومات", "نظم معلومات"),
"General": ("general", "عام", "عامة", "عامه", "foundation", "pre-major", "undeclared", "جنرال"),
}
SEMESTER = r"(?:semester|sem|term|سمستر|السمستر|ترم|الترم|فصل|الفصل)(?:\s+الدراسي)?"
NUMBERS = [
("one", "first", "واحد", "الاول", "اول"),
("two", "second", "اثنين", "الثاني", "ثاني"),
("three", "third", "ثلاثة", "الثالث", "ثالث"),
("four", "fourth", "اربعة", "الرابع", "رابع"),
("five", "fifth", "خمسة", "الخامس", "خامس"),
("six", "sixth", "ستة", "السادس", "سادس"),
("seven", "seventh", "سبعة", "سبع", "السابع", "سابع"),
("eight", "eighth", "ثمانية", "الثامن", "ثامن"),
]
SIGNALS = {
"PREREQUISITE": ("prerequisite", "prerequisites", "prereq", "pre req", "متطلب", "متطلب سابق", "المتطلب السابق", "ايه قبلها", "لازم اخد ايه قبلها", "محتاجة ايه قبلها", "required before"),
"MAJOR_ELECTIVES": ("elective", "electives", "ec", "optional courses", "اختياري", "اختيارية", "اختياريات", "الاختيارية", "اختيارات التخصص"),
"CORE_REQUIRED_COURSES": ("core", "required courses", "required subjects", "mandatory", "الكور", "الاساسية", "اساسية", "المطلوبة", "الاجبارية", "متطلبات التخصص"),
"SPECIALIZATION": ("specialization", "specialisation", "specialize", "specialise", "choose major", "choose a major", "declare major", "select a major", "major eligibility", "تخصص", "اتخصص", "اختار تخصص"),
"BROAD_MAJOR_CURRICULUM": ("curriculum", "study plan", "major courses", "courses for my major", "courses in my major", "courses do i need for my major", "مواد تخصصي", "الخطة الدراسية", "كل مواد التخصص"),
}
SPELLINGS = {"semster": "semester", "electivs": "electives", "electve": "elective", "prerequisit": "prerequisite", "mawad": "مواد", "tarm": "term"}

