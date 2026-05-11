import math


candidateData = {
    "Arjun Sharma"    : "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning",
    "Priya Nair"      : "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS",
    "Rahul Gupta"     : "Java, Spring Boot, MySql, Microservices, Docker, kubernates",
    "Sneha Patel"     : "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib",
    "Vikram Singh"    : "C++, Algoritms, Data Structure, competitive programming, python",
    "Ananya Krishnan" : "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD",
    "Karan Mehta"     : "Python, Sklearn, XGboost, feature engineering, SQL, tableau",
    "Deepika Rao"     : "Java, Android, Kotlin, Firebase, REST, UI/UX, figma",
    "Aditya Kumar"    : "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest",
    "Meera Iyer"      : "python, R, statistics, ML, regression, clustering, Power-BI",
}

jobDescriptions = {
    "JD-1 — Kakao (ML Engineer)" : [
        "python", "machine learning", "deep learning", "tensorflow",
        "pytorch", "sql", "data visualization",
        "nlp", "bert", "feature engineering", "statistics"
    ],
    "JD-2 — Naver (Backend Engineer)" : [
        "java", "spring boot", "mysql", "postgresql",
        "microservices", "docker", "kubernetes",
        "rest api", "ci/cd", "redis"
    ],
    "JD-3 — Line (Frontend Engineer)" : [
        "javascript", "react", "vue", "typescript", "rest api", "html/css",
        "node.js", "graphql", "redux", "jest", "aws"
    ],
}
SKILL_ALIASES = {
# Languages
"python": "python",
"pyhton": "python",
"java": "java",
"javascript": "javascript",
"javascrpit": "javascript",
"js": "javascript",
"typescript": "typescript",
"typescrpit": "typescript",
"c++": "cpp",
"cpp": "cpp",
"r": "r",
"kotlin": "kotlin",
# ML / Data
"machinelearning": "machine_learning",
"machine learning": "machine_learning",
"ml": "machine_learning",
"sklearn": "machine_learning",
"deeplearning": "deep_learning",
"deep learning": "deep_learning",
"deep-learning": "deep_learning",
"tensorflow": "tensorflow",
"pytorch": "pytorch",
"keras": "keras",
"nlp": "nlp",
"bert": "bert",
"xgboost": "xgboost",
"feature engineering": "feature_engineering",
"statistics": "statistics",
"stats": "statistics",
"regression": "regression",
"clustering": "clustering",
"data-viz": "data_visualization",
"data visualization": "data_visualization",
"data viz": "data_visualization",
"matplotlib": "data_visualization",
"tableau": "data_visualization",
"power-bi": "data_visualization",
"power bi": "data_visualization",
"powerbi": "data_visualization",
"pandas": "pandas",
"numpy": "numpy",
# Web — Frontend
"react": "react",
"reacts": "react",
"reactjs": "react",
"vue": "vue",
"vue.js": "vue",
"vuejs": "vue",
"redux": "redux",
"tailwind": "tailwind",
"html/css": "html_css",
"html css": "html_css",
"html": "html_css",
"css": "html_css",
"jest": "jest",
"graphql": "graphql",
# Web — Backend
"node.js": "nodejs",
"nodejs": "nodejs",
"node js": "nodejs",
"flask": "flask",
"spring boot": "spring_boot",
"springboot": "spring_boot",
"rest api": "rest_api",
"rest": "rest_api",
"restapi": "rest_api",
"microservices": "microservices",
# Databases
"sql": "sql",
"mysql": "mysql",
"mysq": "mysql",
"postgresql": "postgresql",
"postgres": "postgresql",
"mongodb": "mongodb",
"redis": "redis",
# DevOps / Cloud
"docker": "docker",
"kubernetes": "kubernetes",
"kubernates": "kubernetes",
"k8s": "kubernetes",
"ci/cd": "ci_cd",
"cicd": "ci_cd",
"ci cd": "ci_cd",
"aws": "aws",
# Mobile
"android": "android",
"firebase": "firebase",
# CS Fundamentals
"algorithms": "algorithms",
"algoritms": "algorithms",
"data structure": "data_structures",
"data structures": "data_structures",
"competitive programming": "competitive_programming",
# Design
"ui/ux": "ui_ux",
"ui ux": "ui_ux",
"figma": "figma",
}



aliasKeys = sorted(SKILL_ALIASES.keys(), key=lambda k: -len(k))

def cleanSkills(raw_text):
    tokens = [chunk.strip().lower() for chunk in raw_text.split(",")]

    seen   = set()
    result = []

    for token in tokens:
        canon = None
        for key in aliasKeys:      # longest keys tried first
            if token == key:
                canon = SKILL_ALIASES[key]
                break

        if canon is None:  # not in alias map → discard
            continue
        if canon in seen:  # duplicate → skip
            continue

        seen.add(canon)
        result.append(canon)

    return result



def build_vocab(all_skills_dict):
    pool = set()
    for skill_list in all_skills_dict.values():
        pool.update(skill_list)
    return sorted(pool)   # alphabetical — required!

def make_tfidf_vectors(all_skills_dict, vocab):
    total = len(all_skills_dict)  # = 10

    df  = {sk: sum(1 for s in all_skills_dict.values() if sk in s) for sk in vocab}
    idf = {sk: math.log(total / df[sk]) for sk in vocab}

    vectors = {}
    for name, skills in all_skills_dict.items():
        N       = len(skills)
        sk_set  = set(skills)
        vec     = []
        for sk in vocab:
            if sk in sk_set:
                vec.append((1/N) * idf[sk])
            else:
                vec.append(0.0)
        vectors[name] = vec
    return vectors

def make_jd_vector(jd_raw_skills, vocab):
    jd_canonical = set()
    for raw in jd_raw_skills:
        key = raw.lower()
        if key in SKILL_ALIASES:
            jd_canonical.add(SKILL_ALIASES[key])
    return [1 if sk in jd_canonical else 0 for sk in vocab]

def cosine_sim(vec_a, vec_b):
    dot    = sum(a*b for a,b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a*a for a in vec_a))
    norm_b = math.sqrt(sum(b*b for b in vec_b))
    if norm_a == 0 or norm_b == 0: return 0.0
    return dot / (norm_a * norm_b)

def top_candidates(tfidf_vecs, jd_vec, n=3):
    scored = []
    for name, vec in tfidf_vecs.items():
        sim = round(cosine_sim(vec, jd_vec), 2)
        scored.append((name, sim))
    scored.sort(key=lambda p: (-p[1], p[0]))  # score desc, name asc on tie
    return scored[:n]

def main():
    cleaned    = {n: cleanSkills(r) for n,r in candidateData.items()}
    vocab      = build_vocab(cleaned)
    tfidf_vecs = make_tfidf_vectors(cleaned, vocab)

    for jd_label, jd_skills in jobDescriptions.items():
        jd_vec  = make_jd_vector(jd_skills, vocab)
        results = top_candidates(tfidf_vecs, jd_vec)
        print(jd_label)
        print(", ".join(f"{n}({s})" for n,s in results))
        print()

if __name__ == "__main__":
    main()