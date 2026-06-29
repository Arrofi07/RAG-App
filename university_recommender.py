# university_recommender.py
#
# Hybrid University Recommender: SQL hard-filters + Vector soft-match
#
# WHY HYBRID?
# ───────────
# A pure vector search ("find universities semantically similar to this query")
# is great for nuanced matching but ignores hard constraints.  If a student
# says "I need a university in Berlin with an English-taught CS Master's under
# €300/semester fees", returning universities in Munich or with German-only
# programs is actively wrong — even if they score highly on semantic similarity.
#
# A pure SQL filter is precise but can't handle vague preferences like
# "strong in robotics" or "good research reputation in AI".
#
# The hybrid approach:
#   Stage 1 — SQL hard-filter: eliminate universities that violate any
#              non-negotiable constraint (city, degree type, language, budget).
#   Stage 2 — Vector soft-match: among the survivors, rank by semantic
#              similarity to the student's free-text preferences and field.
#   Stage 3 — Score fusion: combine the SQL-derived attribute score with the
#              vector similarity to produce a final ranked list.
#
# DATABASE
# ────────
# Universities are stored in the same SQLite file as the chat data (data/chatbot.db),
# in a new `universities` table.  We also store a pre-computed 1024-dim embedding
# per university (a text blob of its key attributes) in a `university_embeddings`
# table.  At query time we load only the filtered subset into memory, which keeps
# RAM usage minimal on the M2 / 8 GB machine.
#
# SEEDING
# ───────
# The seed_universities() function populates the DB with a representative set
# of ~30 German universities.  Call it once from main.py on startup if the
# table is empty.  You can expand the seed data or replace it with a CSV import.
#
# MEMORY CONSTRAINT (M2 / 8 GB)
# ───────────────────────────────
# We intentionally do NOT load all embeddings at startup.  Instead, we:
#   1. SQL-filter first  →  typically reduces ~150 universities to ~10–30
#   2. Load only those embeddings  →  ~10–30 × 1024 floats ≈ < 0.5 MB
# This means the recommender adds essentially zero steady-state RAM overhead.
# The bge-m3 model (already loaded by data_loader.py) is reused for embedding.

import json
import logging
import sqlite3
import struct
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/chatbot.db")


# ─────────────────────────────────────────────────────────────────────────────
# Schema helpers
# ─────────────────────────────────────────────────────────────────────────────

_DDL = """
-- Universities master table
-- Each row is one university / program combination.
-- Keeping (university, program) as the unit lets us have e.g.
--   TU Munich / CS Master's  and  TU Munich / Mechanical Eng. Master's
-- as separate filterable rows.
CREATE TABLE IF NOT EXISTS universities (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Identity
    name             TEXT NOT NULL,          -- "Technical University of Munich"
    short_name       TEXT,                   -- "TUM"
    city             TEXT NOT NULL,          -- "Munich"
    state            TEXT,                   -- "Bavaria"
    url              TEXT,                   -- official program URL
    uni_assist       INTEGER DEFAULT 1,      -- 1 = applies via uni-assist

    -- Program details
    degree_type      TEXT NOT NULL,          -- "Master", "Bachelor", "PhD"
    field            TEXT NOT NULL,          -- "Computer Science"
    program_name     TEXT,                   -- "Informatics (M.Sc.)"
    language         TEXT NOT NULL,          -- "English", "German", "Both"

    -- Hard-filter columns (indexed for fast WHERE clauses)
    semester_fee_eur INTEGER DEFAULT 300,    -- typical range: 150–500 EUR
    german_required  TEXT    DEFAULT "B2",   -- min German level, or "None"
    english_required TEXT    DEFAULT "B2",   -- min English level (IELTS/TOEFL equiv)
    min_gpa          REAL    DEFAULT 0.0,    -- German GPA equiv (1.0=best, 4.0=pass)

    -- Soft-match columns (used to build embedding text)
    research_areas   TEXT,                   -- comma-separated, e.g. "AI, robotics"
    strengths        TEXT,                   -- free text about program strengths
    notes            TEXT,                   -- anything extra

    -- Application window
    winter_deadline  TEXT,                   -- "July 15" or NULL
    summer_deadline  TEXT                    -- "January 15" or NULL
);

CREATE INDEX IF NOT EXISTS idx_uni_city   ON universities(city);
CREATE INDEX IF NOT EXISTS idx_uni_field  ON universities(field);
CREATE INDEX IF NOT EXISTS idx_uni_degree ON universities(degree_type);
CREATE INDEX IF NOT EXISTS idx_uni_lang   ON universities(language);

-- Pre-computed embeddings for vector soft-match
-- Stored as raw IEEE-754 little-endian float32 bytes (4 bytes × 1024 dims).
-- This avoids a JSON parse on every query and is compact (~4 KB per row).
CREATE TABLE IF NOT EXISTS university_embeddings (
    university_id INTEGER PRIMARY KEY REFERENCES universities(id),
    embedding     BLOB NOT NULL        -- 1024 float32 values
);
"""


def _init_schema(con: sqlite3.Connection) -> None:
    """Create tables if they don't exist yet."""
    con.executescript(_DDL)
    con.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Seed data — representative German universities
# ─────────────────────────────────────────────────────────────────────────────
# Each dict maps exactly to the `universities` columns.
# Expand this list or load from a CSV to cover more programs.

SEED_UNIVERSITIES = [
    # ── Top technical universities ────────────────────────────────────────────
    {
        "name": "Technical University of Munich", "short_name": "TUM",
        "city": "Munich", "state": "Bavaria",
        "url": "https://www.tum.de/en/studies/degree-programs",
        "degree_type": "Master", "field": "Computer Science",
        "program_name": "Informatics (M.Sc.)", "language": "English",
        "semester_fee_eur": 144, "german_required": "None", "english_required": "B2",
        "min_gpa": 2.5,
        "research_areas": "AI, machine learning, robotics, security",
        "strengths": "World-class research, strong industry connections (BMW, Siemens), excellent career services",
        "winter_deadline": "May 31", "summer_deadline": "November 30",
    },
    {
        "name": "Technical University of Munich", "short_name": "TUM",
        "city": "Munich", "state": "Bavaria",
        "url": "https://www.tum.de/en/studies/degree-programs",
        "degree_type": "Master", "field": "Electrical Engineering",
        "program_name": "Electrical and Computer Engineering (M.Sc.)", "language": "English",
        "semester_fee_eur": 144, "german_required": "None", "english_required": "B2",
        "min_gpa": 2.5,
        "research_areas": "power systems, embedded systems, signal processing",
        "strengths": "DFG-funded labs, Fraunhofer partnerships, international student body",
        "winter_deadline": "May 31", "summer_deadline": None,
    },
    {
        "name": "RWTH Aachen University", "short_name": "RWTH Aachen",
        "city": "Aachen", "state": "North Rhine-Westphalia",
        "url": "https://www.rwth-aachen.de/cms/root/studium/Im-Studium/~rnj/Studiengaenge/",
        "degree_type": "Master", "field": "Mechanical Engineering",
        "program_name": "Mechanical Engineering (M.Sc.)", "language": "Both",
        "semester_fee_eur": 260, "german_required": "B2", "english_required": "B2",
        "min_gpa": 2.5,
        "research_areas": "manufacturing, automotive, materials science",
        "strengths": "Largest technical university in Germany, automotive industry hub",
        "winter_deadline": "June 15", "summer_deadline": "December 15",
    },
    {
        "name": "Karlsruhe Institute of Technology", "short_name": "KIT",
        "city": "Karlsruhe", "state": "Baden-Württemberg",
        "url": "https://www.kit.edu/english/education.php",
        "degree_type": "Master", "field": "Computer Science",
        "program_name": "Computer Science (M.Sc.)", "language": "Both",
        "semester_fee_eur": 155, "german_required": "B2", "english_required": "B2",
        "min_gpa": 2.5,
        "research_areas": "algorithms, software engineering, data science, cybersecurity",
        "strengths": "Elite research institution, Helmholtz Association member",
        "winter_deadline": "May 15", "summer_deadline": "November 15",
    },
    # ── Strong all-round universities ─────────────────────────────────────────
    {
        "name": "Ludwig Maximilian University of Munich", "short_name": "LMU Munich",
        "city": "Munich", "state": "Bavaria",
        "url": "https://www.en.uni-muenchen.de/students/grad/index.html",
        "degree_type": "Master", "field": "Data Science",
        "program_name": "Data Science (M.Sc.)", "language": "English",
        "semester_fee_eur": 144, "german_required": "None", "english_required": "B2",
        "min_gpa": 2.5,
        "research_areas": "statistics, big data, bioinformatics, NLP",
        "strengths": "Excellence university, close collaboration with TUM, Max Planck Institutes",
        "winter_deadline": "May 15", "summer_deadline": None,
    },
    {
        "name": "Heidelberg University", "short_name": "Heidelberg",
        "city": "Heidelberg", "state": "Baden-Württemberg",
        "url": "https://www.uni-heidelberg.de/en/study",
        "degree_type": "Master", "field": "Biosciences",
        "program_name": "Molecular Biosciences (M.Sc.)", "language": "English",
        "semester_fee_eur": 171, "german_required": "None", "english_required": "B2",
        "min_gpa": 2.3,
        "research_areas": "cancer biology, neuroscience, structural biology",
        "strengths": "Germany's oldest university, DKFZ and EMBL on campus",
        "winter_deadline": "June 15", "summer_deadline": None,
    },
    {
        "name": "Freie Universität Berlin", "short_name": "FU Berlin",
        "city": "Berlin", "state": "Berlin",
        "url": "https://www.fu-berlin.de/en/studium/index.html",
        "degree_type": "Master", "field": "Political Science",
        "program_name": "Global Governance (M.A.)", "language": "English",
        "semester_fee_eur": 316, "german_required": "None", "english_required": "B2",
        "min_gpa": 2.5,
        "research_areas": "international relations, governance, EU policy",
        "strengths": "Strong humanities/social sciences, Berlin policy network",
        "winter_deadline": "May 31", "summer_deadline": None,
    },
    {
        "name": "Humboldt University of Berlin", "short_name": "HU Berlin",
        "city": "Berlin", "state": "Berlin",
        "url": "https://www.hu-berlin.de/en/studies",
        "degree_type": "Master", "field": "Computer Science",
        "program_name": "Computer Science (M.Sc.)", "language": "German",
        "semester_fee_eur": 316, "german_required": "C1", "english_required": "B2",
        "min_gpa": 2.5,
        "research_areas": "theory of computation, AI, bioinformatics",
        "strengths": "Historic excellence university, Berlin tech ecosystem",
        "winter_deadline": "June 15", "summer_deadline": "December 15",
    },
    # ── Applied science universities (HAW/FH) — more accessible ──────────────
    {
        "name": "Hamburg University of Applied Sciences", "short_name": "HAW Hamburg",
        "city": "Hamburg", "state": "Hamburg",
        "url": "https://www.haw-hamburg.de/en/",
        "degree_type": "Master", "field": "Computer Science",
        "program_name": "Applied Computer Science (M.Sc.)", "language": "English",
        "semester_fee_eur": 340, "german_required": "None", "english_required": "B2",
        "min_gpa": 3.0,
        "research_areas": "software engineering, IoT, media technology",
        "strengths": "Industry-focused, strong internship culture, affordable city",
        "winter_deadline": "July 1", "summer_deadline": "January 1",
    },
    {
        "name": "Munich University of Applied Sciences", "short_name": "MUAS / HM",
        "city": "Munich", "state": "Bavaria",
        "url": "https://www.hm.edu/en/",
        "degree_type": "Master", "field": "Business Administration",
        "program_name": "Innovation and Entrepreneurship (M.B.A.)", "language": "English",
        "semester_fee_eur": 144, "german_required": "None", "english_required": "B2",
        "min_gpa": 3.0,
        "research_areas": "entrepreneurship, startup ecosystems, digital transformation",
        "strengths": "Strong industry links, Munich startup scene, affordable fees",
        "winter_deadline": "July 15", "summer_deadline": None,
    },
    {
        "name": "Cologne University of Applied Sciences", "short_name": "TH Köln",
        "city": "Cologne", "state": "North Rhine-Westphalia",
        "url": "https://www.th-koeln.de/en/",
        "degree_type": "Master", "field": "Information Systems",
        "program_name": "Digital Sciences (M.Sc.)", "language": "English",
        "semester_fee_eur": 280, "german_required": "None", "english_required": "B2",
        "min_gpa": 3.0,
        "research_areas": "digital humanities, data management, smart cities",
        "strengths": "Largest Fachhochschule in Germany, very international",
        "winter_deadline": "July 1", "summer_deadline": "January 15",
    },
    # ── More cities / fields ──────────────────────────────────────────────────
    {
        "name": "University of Stuttgart", "short_name": "UniStuttgart",
        "city": "Stuttgart", "state": "Baden-Württemberg",
        "url": "https://www.uni-stuttgart.de/en/study/",
        "degree_type": "Master", "field": "Aerospace Engineering",
        "program_name": "Aerospace Engineering (M.Sc.)", "language": "Both",
        "semester_fee_eur": 192, "german_required": "B2", "english_required": "B2",
        "min_gpa": 2.5,
        "research_areas": "aerodynamics, spacecraft, propulsion",
        "strengths": "Heart of German automotive and aerospace industry",
        "winter_deadline": "June 15", "summer_deadline": None,
    },
    {
        "name": "Technical University of Berlin", "short_name": "TU Berlin",
        "city": "Berlin", "state": "Berlin",
        "url": "https://www.tu.berlin/en/studying/",
        "degree_type": "Master", "field": "Computer Science",
        "program_name": "Computer Engineering (M.Sc.)", "language": "English",
        "semester_fee_eur": 316, "german_required": "None", "english_required": "B2",
        "min_gpa": 2.5,
        "research_areas": "distributed systems, computer vision, HCI",
        "strengths": "International Master's programs, Berlin tech ecosystem, strong alumni",
        "winter_deadline": "May 31", "summer_deadline": None,
    },
    {
        "name": "University of Frankfurt", "short_name": "Goethe Uni Frankfurt",
        "city": "Frankfurt", "state": "Hesse",
        "url": "https://www.uni-frankfurt.de/en",
        "degree_type": "Master", "field": "Finance",
        "program_name": "Finance (M.Sc.)", "language": "English",
        "semester_fee_eur": 316, "german_required": "None", "english_required": "B2",
        "min_gpa": 2.5,
        "research_areas": "financial markets, banking, quantitative finance",
        "strengths": "European banking capital, ECB and Deutsche Bank partnerships",
        "winter_deadline": "May 15", "summer_deadline": None,
    },
    {
        "name": "University of Hamburg", "short_name": "UHH",
        "city": "Hamburg", "state": "Hamburg",
        "url": "https://www.uni-hamburg.de/en/studium.html",
        "degree_type": "Master", "field": "Environmental Science",
        "program_name": "Climate and Earth System Sciences (M.Sc.)", "language": "English",
        "semester_fee_eur": 340, "german_required": "None", "english_required": "B2",
        "min_gpa": 2.5,
        "research_areas": "climate modelling, oceanography, sustainability",
        "strengths": "Port city, strong earth science tradition, DKRZ supercomputer access",
        "winter_deadline": "July 1", "summer_deadline": None,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# UniversityRecommender
# ─────────────────────────────────────────────────────────────────────────────

class UniversityRecommender:
    """
    Hybrid recommender combining SQL hard-filters with vector soft-match.

    Usage (in main.py):
        recommender = UniversityRecommender()
        recommender.ensure_seeded()          # populate DB on first run
        result = recommender.recommend(profile, question)
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    # ── Connection helper ─────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        """
        Return a new SQLite connection.
        We open/close per-query rather than keeping a persistent connection
        to avoid threading issues (FastAPI runs handlers in threads).
        """
        con = sqlite3.connect(self.db_path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _ensure_schema(self) -> None:
        con = self._conn()
        try:
            _init_schema(con)
        finally:
            con.close()

    # ── Seed data ─────────────────────────────────────────────────────────────

    def ensure_seeded(self, embed_fn=None) -> None:
        """
        Populate the universities table if it's empty.

        Args:
            embed_fn : callable(list[str]) → {"dense": list[list[float]], ...}
                       If provided, we also compute and store embeddings.
                       If None, embeddings are skipped and only SQL filtering works
                       (vector soft-match is disabled until embeddings are computed).
        """
        con = self._conn()
        try:
            count = con.execute("SELECT COUNT(*) FROM universities").fetchone()[0]
            if count > 0:
                log.info("University DB already seeded (%d rows)", count)
                return

            log.info("Seeding university database with %d entries…", len(SEED_UNIVERSITIES))

            for uni in SEED_UNIVERSITIES:
                cur = con.execute(
                    """INSERT INTO universities
                       (name, short_name, city, state, url, degree_type, field,
                        program_name, language, semester_fee_eur, german_required,
                        english_required, min_gpa, research_areas, strengths, notes,
                        winter_deadline, summer_deadline)
                       VALUES
                       (:name, :short_name, :city, :state, :url, :degree_type, :field,
                        :program_name, :language, :semester_fee_eur, :german_required,
                        :english_required, :min_gpa, :research_areas, :strengths,
                        :notes, :winter_deadline, :summer_deadline)""",
                    {
                        "notes": uni.get("notes", ""),
                        **uni,
                    },
                )
                uni_id = cur.lastrowid

                # Store embedding if embed_fn is available
                if embed_fn is not None:
                    embed_text = _build_embed_text(uni)
                    try:
                        result = embed_fn([embed_text])
                        vec    = result["dense"][0]          # list of 1024 floats
                        blob   = _floats_to_blob(vec)        # compact binary storage
                        con.execute(
                            "INSERT INTO university_embeddings (university_id, embedding) VALUES (?, ?)",
                            (uni_id, blob),
                        )
                    except Exception as e:
                        log.warning("Embedding failed for '%s': %s", uni["name"], e)

            con.commit()
            log.info("✅ University DB seeded.")
        finally:
            con.close()

    def compute_missing_embeddings(self, embed_fn) -> int:
        """
        Compute and store embeddings for any universities that don't have one yet.
        Call this after ensure_seeded() if you want vector soft-match to work.
        Returns the number of newly computed embeddings.
        """
        con = self._conn()
        try:
            # Find universities without embeddings
            rows = con.execute(
                """SELECT u.id, u.name, u.field, u.research_areas, u.strengths,
                          u.degree_type, u.language, u.city
                   FROM universities u
                   LEFT JOIN university_embeddings e ON u.id = e.university_id
                   WHERE e.university_id IS NULL"""
            ).fetchall()

            if not rows:
                return 0

            log.info("Computing embeddings for %d universities…", len(rows))
            count = 0

            for row in rows:
                uni_dict = dict(row)
                embed_text = _build_embed_text(uni_dict)
                try:
                    result = embed_fn([embed_text])
                    vec    = result["dense"][0]
                    blob   = _floats_to_blob(vec)
                    con.execute(
                        "INSERT OR REPLACE INTO university_embeddings (university_id, embedding) VALUES (?, ?)",
                        (uni_dict["id"], blob),
                    )
                    count += 1
                except Exception as e:
                    log.warning("Embedding failed for '%s': %s", uni_dict["name"], e)

            con.commit()
            log.info("✅ Computed %d new university embeddings.", count)
            return count
        finally:
            con.close()

    # ── Stage 1: SQL hard-filter ──────────────────────────────────────────────

    def _sql_filter(self, profile: dict) -> list[dict]:
        """
        Filter universities using the student's hard constraints.

        Hard constraints applied:
        - degree_type  : must match (Master/Bachelor/PhD)
        - field        : fuzzy LIKE match on the student's field_of_study
        - city         : if student specified preferred cities, only those
        - language     : if German level < B2, only English-taught programs
        - semester_fee : must be ≤ budget converted to semester fee equivalent
        - german_required: if student has no German, exclude German-only programs

        Returns a list of university row dicts passing all filters.
        """
        con = self._conn()
        try:
            # ── Build WHERE clauses dynamically ──────────────────────────────
            conditions = []
            params     = []

            # Degree type filter
            target_degree = profile.get("target_degree", "")
            if "master" in target_degree.lower():
                conditions.append("degree_type = 'Master'")
            elif "bachelor" in target_degree.lower():
                conditions.append("degree_type = 'Bachelor'")
            elif "phd" in target_degree.lower() or "doctor" in target_degree.lower():
                conditions.append("degree_type = 'PhD'")

            # Field of study — flexible LIKE match so "Computer Science" matches
            # "Computer Science", "Applied Computer Science", etc.
            field = profile.get("field_of_study", "")
            if field:
                # Split multi-word field into terms and OR-match each
                terms = [t.strip() for t in field.replace(",", " ").split() if len(t) > 2]
                if terms:
                    field_clauses = " OR ".join("field LIKE ?" for _ in terms)
                    conditions.append(f"({field_clauses})")
                    params.extend(f"%{t}%" for t in terms)

            # City filter — only if student specified preferred cities
            cities_raw = profile.get("target_cities", "")
            if cities_raw:
                cities = [c.strip() for c in cities_raw.split(",") if c.strip()]
                if cities:
                    city_placeholders = ",".join("?" for _ in cities)
                    conditions.append(f"city IN ({city_placeholders})")
                    params.extend(cities)

            # Language filter — if student has little/no German, exclude German-only
            german_level = profile.get("german_level", "None (complete beginner)")
            # Map German level strings to a numeric scale for comparison
            german_scale = {
                "none (complete beginner)": 0, "a1": 1, "a2": 2,
                "b1": 3, "b2": 4, "c1": 5, "c2 (native-like)": 6
            }
            student_german = german_scale.get(german_level.lower(), 0)
            if student_german < 4:  # below B2
                conditions.append("language != 'German'")

            # Budget filter — convert monthly budget to semester fee headroom
            # Rule of thumb: semester fee should not exceed ~30% of one month's budget
            budget = profile.get("budget_monthly_eur", 1000)
            if budget:
                try:
                    max_fee = int(budget) * 0.5  # allow up to 50% of monthly budget
                    conditions.append("semester_fee_eur <= ?")
                    params.append(int(max_fee))
                except (ValueError, TypeError):
                    pass  # skip budget filter if value is invalid

            # ── Execute query ─────────────────────────────────────────────────
            where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            sql   = f"""
                SELECT u.*, e.embedding
                FROM universities u
                LEFT JOIN university_embeddings e ON u.id = e.university_id
                {where}
                LIMIT 50
            """

            rows = con.execute(sql, params).fetchall()
            result = [dict(r) for r in rows]
            log.info("SQL hard-filter: %d universities passed", len(result))
            return result

        finally:
            con.close()

    # ── Stage 2: Vector soft-match ────────────────────────────────────────────

    def _vector_rank(
        self,
        candidates: list[dict],
        query_vec:  list[float],
    ) -> list[dict]:
        """
        Re-rank SQL-filtered candidates by cosine similarity to the query vector.

        Adds a `vector_score` float (0–1) to each candidate dict.
        Universities without a stored embedding get a neutral score of 0.5
        so they're not unfairly penalised.
        """
        if not query_vec:
            # No embedding available; return candidates unchanged with neutral score
            for c in candidates:
                c["vector_score"] = 0.5
            return candidates

        q = np.array(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            for c in candidates:
                c["vector_score"] = 0.5
            return candidates

        q_unit = q / q_norm  # unit vector for cosine similarity

        for candidate in candidates:
            blob = candidate.get("embedding")
            if blob:
                try:
                    vec   = _blob_to_floats(blob)      # decode stored embedding
                    v     = np.array(vec, dtype=np.float32)
                    v_norm = np.linalg.norm(v)
                    if v_norm > 0:
                        # Cosine similarity = dot product of unit vectors
                        candidate["vector_score"] = float(np.dot(q_unit, v / v_norm))
                    else:
                        candidate["vector_score"] = 0.5
                except Exception:
                    candidate["vector_score"] = 0.5
            else:
                candidate["vector_score"] = 0.5   # no embedding → neutral

        return candidates

    # ── Stage 3: Score fusion ─────────────────────────────────────────────────

    def _fuse_and_rank(
        self,
        candidates: list[dict],
        profile:    dict,
        top_k:      int = 5,
    ) -> list[dict]:
        """
        Combine vector similarity with attribute-based bonus scores to produce
        a final match score, then return the top_k ranked universities.

        Attribute bonuses reward universities that match soft preferences:
          +0.05 each for matching a preferred city
          +0.05 if the program is fully in English and student has no German
          +0.03 if the university also uses uni-assist (consistent with student's workflow)
        These are small nudges — the vector score drives the main ranking.
        """
        preferred_cities = [
            c.strip().lower()
            for c in profile.get("target_cities", "").split(",")
            if c.strip()
        ]
        german_level  = profile.get("german_level", "none").lower()
        no_german     = german_level in ("none (complete beginner)", "a1", "a2")

        for c in candidates:
            score  = c.get("vector_score", 0.5)
            bonus  = 0.0

            # City preference bonus
            if preferred_cities and c.get("city", "").lower() in preferred_cities:
                bonus += 0.05

            # English-only bonus for students without German
            if no_german and c.get("language") == "English":
                bonus += 0.05

            # uni-assist bonus (student already knows the portal)
            if c.get("uni_assist"):
                bonus += 0.02

            c["match_score"] = min(score + bonus, 1.0)  # cap at 100%

            # Human-readable reason string
            reasons = []
            if c.get("field"):
                reasons.append(f"Field matches: {c['field']}")
            if c.get("language") == "English" and no_german:
                reasons.append("English-taught (no German required)")
            if preferred_cities and c.get("city", "").lower() in preferred_cities:
                reasons.append(f"Located in your preferred city: {c['city']}")
            if c.get("research_areas"):
                reasons.append(f"Research areas: {c['research_areas']}")
            c["match_reason"] = " | ".join(reasons) if reasons else ""

        # Sort by match_score descending
        ranked = sorted(candidates, key=lambda x: x["match_score"], reverse=True)
        return ranked[:top_k]

    # ── Public entry point ────────────────────────────────────────────────────

    def recommend(
        self,
        profile:   dict,
        question:  str   = "",
        embed_fn         = None,
        top_k:     int   = 5,
    ) -> dict:
        """
        Run the full hybrid recommendation pipeline and return structured results.

        Args:
            profile  : student's advisor profile dict (from storage.py)
            question : the student's free-text question (used to build query vector)
            embed_fn : callable(list[str]) → {"dense": [...], "sparse": [...]}
                       Pass data_loader.embed to enable vector soft-match.
                       If None, only SQL filtering + attribute bonuses are used.
            top_k    : number of recommendations to return

        Returns:
            {
                "universities": [
                    {
                        "name": str, "city": str, "degree_type": str,
                        "field": str, "language": str, "semester_fee_eur": int,
                        "url": str, "match_score": float (0–1),
                        "match_reason": str,
                        "winter_deadline": str | None,
                        "summer_deadline": str | None,
                    },
                    ...
                ],
                "total_filtered": int,   # how many passed SQL hard-filter
                "profile_used":   dict,  # the profile dict that drove filtering
            }
        """
        # Stage 1 — SQL hard-filter (always runs)
        candidates = self._sql_filter(profile)

        if not candidates:
            # No hard-filter matches — return empty rather than misleading results
            return {
                "universities": [],
                "total_filtered": 0,
                "profile_used": profile,
            }

        # Stage 2 — Vector soft-match (runs only if embed_fn is available)
        query_vec = []
        if embed_fn is not None:
            # Build a rich query text from the student's profile and question
            query_text = _build_query_text(profile, question)
            try:
                result    = embed_fn([query_text])
                query_vec = result["dense"][0]
            except Exception as e:
                log.warning("Query embedding failed: %s — skipping vector ranking", e)

        candidates = self._vector_rank(candidates, query_vec)

        # Stage 3 — Score fusion and top-k selection
        ranked = self._fuse_and_rank(candidates, profile, top_k=top_k)

        # Clean up: remove internal blob column before returning
        for u in ranked:
            u.pop("embedding", None)

        return {
            "universities":   ranked,
            "total_filtered": len(candidates),
            "profile_used":   profile,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _build_embed_text(uni: dict) -> str:
    """
    Build a rich text description of a university for embedding.

    The text combines all soft-match attributes into natural-language sentences
    so the embedding captures the university's academic identity, not just
    its name.  Structured data (fees, deadlines) is excluded because those
    are handled by SQL hard-filters — including them here would confuse the
    semantic similarity calculation.
    """
    parts = [
        f"{uni.get('name', '')} ({uni.get('short_name', '')})",
        f"Located in {uni.get('city', '')}.",
        f"Offers {uni.get('degree_type', '')} in {uni.get('field', '')}.",
        f"Program: {uni.get('program_name', '')}.",
        f"Language of instruction: {uni.get('language', '')}.",
    ]
    if uni.get("research_areas"):
        parts.append(f"Research areas: {uni['research_areas']}.")
    if uni.get("strengths"):
        parts.append(f"Strengths: {uni['strengths']}.")
    if uni.get("notes"):
        parts.append(uni["notes"])

    return " ".join(parts)


def _build_query_text(profile: dict, question: str) -> str:
    """
    Build a query text from the student's profile for embedding.
    This is matched against university embeddings to find semantic similarity.
    """
    parts = []

    if profile.get("field_of_study"):
        parts.append(f"Studying {profile['field_of_study']}.")
    if profile.get("target_degree"):
        parts.append(f"Seeking a {profile['target_degree']} degree.")
    if profile.get("target_cities"):
        parts.append(f"Preferred cities: {profile['target_cities']}.")
    if profile.get("german_level"):
        parts.append(f"German language level: {profile['german_level']}.")
    if profile.get("extra_notes"):
        parts.append(profile["extra_notes"])
    if question:
        parts.append(question)

    return " ".join(parts) if parts else "General university recommendation Germany"


def _floats_to_blob(floats: list) -> bytes:
    """Pack a list of float32 values into compact binary bytes for SQLite BLOB storage."""
    # struct.pack with format 'f' × N produces 4 bytes per float (IEEE 754)
    return struct.pack(f"{len(floats)}f", *floats)


def _blob_to_floats(blob: bytes) -> list:
    """Unpack a binary BLOB back into a list of float32 values."""
    n = len(blob) // 4   # each float32 is 4 bytes
    return list(struct.unpack(f"{n}f", blob))