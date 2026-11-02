"""
Reference answers for the answer_correctness metric in evaluate.py.
Written by hand from the same source documents as test_set.py — one entry
per question that has a real (non-negative-test) ground truth.

Keyed by question text so it stays paired with test_set.py without needing
matching list indices.
"""

reference_answers = {
    "What is the case fatality rate of Nipah virus?":
        "The case fatality rate of Nipah virus is estimated at 40% to 75%.",

    "What animal is the natural host of Nipah virus?":
        "Fruit bats of the Pteropodidae family, particularly the Pteropus genus, are the natural host of Nipah virus.",

    "What is the incubation period for Nipah virus infection?":
        "The incubation period ranges from 3 to 14 days, though incubation of up to 45 days has been reported in rare cases.",

    "Is there an approved treatment or vaccine for Nipah virus?":
        "No. There are currently no approved drugs or vaccines for Nipah virus infection. Treatment is limited to supportive care.",

    "Where and when was Nipah virus first identified?":
        "Nipah virus was first identified during an outbreak among pig farmers in the village of Sungai Nipah, Malaysia, in 1998-1999.",

    "What genus and family does Nipah virus belong to?":
        "Nipah virus belongs to the Henipavirus genus within the Paramyxoviridae family.",

    "What biosafety level is required to work with live Nipah virus for propagation and isolation?":
        "Virus propagation, isolation, quantification and neutralisation of Nipah virus require BSL-4 (biosafety level 4) facilities.",

    "What are the two genetic clades of Nipah virus and where is each found?":
        "Nipah virus strains are divided into two clades: Clade I (NiV-B), found in Bangladesh and India, and Clade II (NiV-M), found in Malaysia.",

    "What proportion of Nipah virus infections are asymptomatic?":
        "Up to 11% of infected individuals remain asymptomatic, though this proportion varies by NiV strain.",

    "How many confirmed Nipah virus cases were reported in West Bengal in January 2026, and who were the patients?":
        "Two laboratory-confirmed Nipah virus cases were reported in West Bengal in January 2026. Both patients were healthcare workers at the same private hospital in Barasat, North 24 Parganas district.",

    "How many contacts were traced and tested during the January 2026 West Bengal Nipah outbreak, and what were the results?":
        "Over 190 contacts were identified and tested during the January 2026 West Bengal outbreak, and all tested negative for Nipah virus.",

    "Which laboratory confirmed the January 2026 West Bengal Nipah cases?":
        "The National Institute of Virology in Pune confirmed the January 2026 West Bengal Nipah virus cases.",

    "What is WHO's risk assessment for the January 2026 Nipah outbreak in India at the global level?":
        "WHO assessed the public health risk posed by the January 2026 Nipah outbreak in India to be low at the global level.",

    "How many cases of Nipah virus human infection were reported between 1998 and 2015?":
        "More than 600 cases of Nipah virus human infection were reported between 1998 and 2015.",

    "When was the Nipah virus R&D Roadmap published?":
        "The Nipah virus R&D Roadmap was published on 10 January 2019.",

    "What is the purpose of the Nipah virus R&D Roadmap?":
        "The roadmap's purpose is to provide a 5-year framework for identifying the vision, strategic goals, and priority areas and activities for accelerating the collaborative development of medical countermeasures — diagnostics, therapeutics, and vaccines — against Nipah virus infection.",

    "What was the case-fatality rate in the original 1998-1999 Nipah outbreak in Malaysia?":
        "The overall case-fatality rate in the original 1998-1999 Nipah outbreak in Malaysia was 40%.",

    "How many Nipah virus outbreaks have occurred in India since 2001, and in which states?":
        "Since 2001, there have been six outbreaks of Nipah virus infection in India: four in the state of Kerala and two in the state of West Bengal.",

    "What was the most severe Nipah outbreak in Kerala, and when did it occur?":
        "The most severe Nipah outbreak in Kerala occurred in May 2018 in Kozhikode, resulting in 23 cases including 21 deaths.",

    # Negative test case ("What is the current price of a Nipah virus vaccine?")
    # deliberately has no reference answer — there is no vaccine, so there is
    # no correct answer to score against.
}
