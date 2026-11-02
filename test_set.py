"""
Ground-truth test set for the Nipah RAG chatbot, built by hand from the six
source documents in `Preprocessed Files/`. Each question is labeled with
`source_file` (which document the answer lives in) and `relevant_text_contains`
(a distinctive substring that should appear in whatever chunk correctly
answers the question).

Labeling by text snippet rather than chunk index is deliberate: LlamaIndex's
MarkdownNodeParser + SentenceSplitter produces different chunk boundaries
than a raw character splitter would, and boundaries can also shift if you
change chunk_size later. Snippet-based labels stay valid regardless of how
the corpus gets re-chunked.

Coverage: all 6 source documents get at least one question, spanning basic
facts (fact sheet), pathogen biology (ECDC disease info), outbreak history
(India DON), R&D landscape (blueprint roadmap), preparedness guidance
(readiness technical brief), and one confirmed no-answer-exists case.
"""

test_set = [
    # --- Nipah_virus_fact_sheet.md ---
    {
        "question": "What is the case fatality rate of Nipah virus?",
        "source_file": "Nipah_virus_fact_sheet.md",
        "relevant_text_contains": ["case fatality rate is estimated at 40% to 75%"],
    },
    {
        "question": "What animal is the natural host of Nipah virus?",
        "source_file": "Nipah_virus_fact_sheet.md",
        "relevant_text_contains": ["Fruit bats of the Pteropodidae family are the natural host of Nipah virus"],
    },
    {
        "question": "What is the incubation period for Nipah virus infection?",
        "source_file": "Nipah_virus_fact_sheet.md",
        "relevant_text_contains": ["incubation period", "ranges from 3 to 14 days"],
    },
    {
        "question": "Is there an approved treatment or vaccine for Nipah virus?",
        "source_file": "Nipah_virus_fact_sheet.md",
        "relevant_text_contains": ["no specific treatments for Nipah", "no currently no approved drugs or vaccines for Nipah virus infection"],
    },

    # --- Disease_information_on_Nipah_virus_disease.md (ECDC) ---
    {
        "question": "Where and when was Nipah virus first identified?",
        "source_file": "Disease_information_on_Nipah_virus_disease.md",
        "relevant_text_contains": ["first reported outbreak occurred in 1998-1999 and involved pig farmers in the village of Sungai Nipah"],
    },
    {
        "question": "What genus and family does Nipah virus belong to?",
        "source_file": "Disease_information_on_Nipah_virus_disease.md",
        "relevant_text_contains": ["NiV belongs to the Henipavirus genus within the Paramyxoviridae family"],
    },
    {
        "question": "What biosafety level is required to work with live Nipah virus for propagation and isolation?",
        "source_file": "Disease_information_on_Nipah_virus_disease.md",
        "relevant_text_contains": ["Virus propagation, isolation, quantification and neutralisation require BSL-4 facilities"],
    },
    {
        "question": "What are the two genetic clades of Nipah virus and where is each found?",
        "source_file": "Disease_information_on_Nipah_virus_disease.md",
        "relevant_text_contains": ["Clade I: NiV-B in Bangladesh and India, and Clade II: NiV-M in Malaysia"],
    },
    {
        "question": "What proportion of Nipah virus infections are asymptomatic?",
        "source_file": "Disease_information_on_Nipah_virus_disease.md",
        "relevant_text_contains": ["Up to 11% of infected individuals remain asymptomatic"],
    },

    # --- Nipah_virus_disease_-_India.md (WHO Disease Outbreak News, Jan 2026) ---
    {
        "question": "How many confirmed Nipah virus cases were reported in West Bengal in January 2026, and who were the patients?",
        "source_file": "Nipah_virus_disease_-_India.md",
        "relevant_text_contains": ["two laboratory-confirmed cases of Nipah virus (NiV) infection in West Bengal State", "Both are healthcare workers at the same private hospital in Barasat"],
    },
    {
        "question": "How many contacts were traced and tested during the January 2026 West Bengal Nipah outbreak, and what were the results?",
        "source_file": "Nipah_virus_disease_-_India.md",
        "relevant_text_contains": ["identified and tested over 190 contacts, who all tested negative for NiV"],
    },
    {
        "question": "Which laboratory confirmed the January 2026 West Bengal Nipah cases?",
        "source_file": "Nipah_virus_disease_-_India.md",
        "relevant_text_contains": ["NiV infection was confirmed at the National Institute of Virology in Pune"],
    },
    {
        "question": "What is WHO's risk assessment for the January 2026 Nipah outbreak in India at the global level?",
        "source_file": "Nipah_virus_disease_-_India.md",
        "relevant_text_contains": ["WHO assesses the public health risk posed by NiV at the global level to be low"],
    },

    # --- nipah___Henipaviruses.md (WHO R&D Blueprint page) ---
    {
        "question": "How many cases of Nipah virus human infection were reported between 1998 and 2015?",
        "source_file": "nipah___Henipaviruses.md",
        "relevant_text_contains": ["From 1998 to 2015, more than 600 cases of Nipah virus human infections were reported"],
    },
    {
        "question": "When was the Nipah virus R&D Roadmap published?",
        "source_file": "nipah___Henipaviruses.md",
        "relevant_text_contains": ["10 JANUARY 2019"],
    },

    # --- nipah_rdblueprint_roadmap_advanceddraftoct2019.md (R&D roadmap, harder — long doc) ---
    {
        "question": "What is the purpose of the Nipah virus R&D Roadmap?",
        "source_file": "nipah_rdblueprint_roadmap_advanceddraftoct2019.md",
        "relevant_text_contains": ["to provide a 5-year framework for identifying"],
    },
    {
        "question": "What was the case-fatality rate in the original 1998-1999 Nipah outbreak in Malaysia?",
        "source_file": "nipah_rdblueprint_roadmap_advanceddraftoct2019.md",
        "relevant_text_contains": ["with an overall case- fatality rate of 40%"],
    },

    # --- Enhancing_readiness_for_a_Nipah_virus_event... .md ---
    {
        "question": "How many Nipah virus outbreaks have occurred in India since 2001, and in which states?",
        "source_file": "Enhancing_readiness_for_a_Nipah_virus_event_in_countries_not_reporting_a_Nipah_virus_event.md",
        "relevant_text_contains": ["Since 2001 there have been six outbreaks of NiV infection in India, four of which were reported in the state of Kerala and two in the state of West Bengal"],
    },
    {
        "question": "What was the most severe Nipah outbreak in Kerala, and when did it occur?",
        "source_file": "Enhancing_readiness_for_a_Nipah_virus_event_in_countries_not_reporting_a_Nipah_virus_event.md",
        "relevant_text_contains": ["The most severe outbreak was reported in May 2018 in Kozhikode which resulted in 23 cases including 21 deaths"],
    },

    # --- Deliberate no-answer-in-corpus case (tests honest retrieval failure) ---
    {
        "question": "What is the current price of a Nipah virus vaccine?",
        "source_file": None,
        "relevant_text_contains": [],  # no document in this corpus has this — there is no licensed vaccine at all
    },
]
