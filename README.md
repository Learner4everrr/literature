# My paper list.

[Hallucination](#Hallucination),
[Review](#Review),
[Classics](#Classical),
[Models](#Models),
[Datasets](#Datasets),
[Prompt](#prompt)
[Evaluation](#Evaluation),
[RAG](#RAG),
[fast_inference, early exiting](#fast_inference),
[Grouppapers](#grouppapers),
[Uncategorized](#Uncategorized),
[Instruction Tuning](#Instruction_Tuning),
[Multi-task Learning](#MTL),
[Detection](#Detection),
[Federated Learning](#FL),
[KV Cache](#KVCache),
[Diffusionmodels](#Diffusionmodels),
[PEFT](#PEFT),
[multimodal](#multimodal),
[distil](#distil)
[QA](#QA)
[RAG+MLLM](#RAG+MLLM)
[cohort study](#Cohort_study)
[Selective classification](#selective_classification)
[model merge](#model_merge)
[Quantization](#quantization)
[complication](#complication)
[Agent](#Agent)

## Some paper list
 - [https://github.com/DengBoCong/nlp-paper?tab=readme-ov-file](https://github.com/DengBoCong/nlp-paper?tab=readme-ov-file)
 - [https://scholar.google.com/citations?user=E0iCaa4AAAAJ&hl=en&oi=sra](https://scholar.google.com/citations?user=E0iCaa4AAAAJ&hl=en&oi=sra)
 - [https://scholar.google.com/citations?hl=en&view_op=search_authors&mauthors=label:natural_language_processing](https://scholar.google.com/citations?hl=en&view_op=search_authors&mauthors=label:natural_language_processing)
 - [paper list](https://docs.google.com/document/d/1BXuF4QfBtzoPksLsTCQBVDyhrCDw3pFRLbvc-G0AgB0/edit)
 - **BioTABQA: Instruction Learning for Biomedical Table Question Answering**. Man Luo et.al. **arxiv**, **2022**, **Number of Citations: **None, ([pdf](./Papers/BioTABQA_Instruction_Learning_for_Biomedical_Table_Question_Answering.pdf))([link](http://arxiv.org/abs/2207.02419v1)).
 - **Large language models encode clinical knowledge**. Singhal Karan et.al. **Nature**, **2023-7-12**, **Number of Citations: **524, ([pdf](./Papers/Large_language_models_encode_clinical_knowledge.pdf))([link](http://dx.doi.org/10.1038/s41586-023-06291-2)).



## <a id="Hallucination">Hallucination</a>
 - **Cognitive Mirage: A Review of Hallucinations in Large Language Models**. Hongbin Ye et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/Cognitive_Mirage_A_Review_of_Hallucinations_in_Large_Language_Models.pdf))([link](http://arxiv.org/abs/2309.06794v1)).
   - Taxonomy of Hallucination
     - Question and Answer
     - Dialog system
     - Summarization system
     - Knowledge Graph with LLMs
     - Cross-model system
   - Detection
     - Classifier
     - Uncertainty Metric: 1)ASTSN: logit output values in their prediction response. 2)BARTSCORE 3)KoK 4)SLAG 5)KLD 6) POLAR
     - Self-Evaluation
     - Evidence Retrieval
 - **A Survey of Hallucination in Large Foundation Models**. Vipula Rawte et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/A_Survey_of_Hallucination_in_Large_Foundation_Models.pdf))([link](http://arxiv.org/abs/2309.05922v1)).
   - Dataset:Med-HALT (Medical Domain Hallucination Test)
 - **Med-HALT: Medical Domain Hallucination Test for Large Language Models**. Ankit Pal et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/Med-HALT_Medical_Domain_Hallucination_Test_for_Large_Language_Models.pdf))([link](http://arxiv.org/abs/2307.15343v2)).
   - [https://medhalt.github.io/](https://medhalt.github.io/)
   - Baseline Models: GPT-3.5 Turbo, Falcon(Penedoetal.,2023b), MPT(MosaicML,2023) and Llama-2(Touvronetal.,2023).
   - Evaluation matrices: Accuracy, **PointwiseScore**



## <a id="Classical">Classical</a>
 - **A unified architecture for natural language processing**. Collobert Ronan et.al. **No journal**, **2008**, **Number of Citations: **2448, ([pdf](./Papers/A_unified_architecture_for_natural_language_processing.pdf))([link](http://dx.doi.org/10.1145/1390156.1390177)).
 - **Imagenet classification with deep convolutional neural networks**. A Krizhevsky et.al. **Commun. ACM**, **2012**, **Number of Citations: **127575, ([pdf](./Papers/Imagenet_classification_with_deep_convolutional_neural_networks.pdf))([link](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)).
 - **Deep residual learning for image recognition**. K He et.al. **CoDIT**, **2016**, **Number of Citations: **211374, ([pdf](./Papers/Deep_residual_learning_for_image_recognition.pdf))([link](http://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)).
 - **Bleu: a method for automatic evaluation of machine translation**. K Papineni et.al. **ACL**, **2002**, **Number of Citations: **26918, ([pdf](./Papers/Bleu_a_method_for_automatic_evaluation_of_machine_translation.pdf))([link](https://aclanthology.org/P02-1040.pdf)).
 - **Attention is all you need**. A Vaswani et.al. **IEEE Signal Process. Lett.**, **2017**, **Number of Citations: **115085, ([pdf](./Papers/Attention_is_all_you_need.pdf))([link](https://proceedings.neurips.cc/paper/7181-attention-is-all)).
 - **A gentle introduction to graph nerual netwroks**. ([Link](https://distill.pub/2021/gnn-intro/))
 - BERT **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. Jacob Devlin et.al. **arxiv**, **2018**, **Number of Citations: **None, ([pdf](./Papers/BERT_Pre-training_of_Deep_Bidirectional_Transformers_for_Language_Understanding.pdf))([link](http://arxiv.org/abs/1810.04805v2)).
 - **TinyBERT: Distilling BERT for Natural Language Understanding**. Xiaoqi Jiao et.al. **arxiv**, **2019**, **Number of Citations: **None, ([pdf](./Papers/TinyBERT_Distilling_BERT_for_Natural_Language_Understanding.pdf))([link](http://arxiv.org/abs/1909.10351v5)).
 - ViT **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**. Alexey Dosovitskiy et.al. **arxiv**, **2020**, **Number of Citations: **None, ([pdf](./Papers/An_Image_is_Worth_16x16_Words_Transformers_for_Image_Recognition_at_Scale.pdf))([link](http://arxiv.org/abs/2010.11929v2)).
 - MAE **Masked Autoencoders Are Scalable Vision Learners**. Kaiming He et.al. **arxiv**, **2021**, **Number of Citations: **None, ([pdf](./Papers/Masked_Autoencoders_Are_Scalable_Vision_Learners.pdf))([link](http://arxiv.org/abs/2111.06377v3)).
 - **Advancing mathematics by guiding human intuition with AI**. Davies Alex et.al. **Nature**, **2021-12-1**, **Number of Citations: **169, ([pdf](./Papers/Advancing_mathematics_by_guiding_human_intuition_with_AI.pdf))([link](http://dx.doi.org/10.1038/s41586-021-04086-x)).
 - GPT **Improving language understanding by generative pre-training**. A Radford et.al. **NA**, **2018**, **Number of Citations: **9162, ([pdf](./Papers/Improving_language_understanding_by_generative_pre-training.pdf))([link](https://www.mikecaptain.com/resources/pdf/GPT-1.pdf)).
 - GPT2 **Language models are unsupervised multitask learners**. A Radford et.al. **OpenAI**, **2019**, **Number of Citations: **9784, ([pdf](./Papers/Language_models_are_unsupervised_multitask_learners.pdf))([link](https://insightcivic.s3.us-east-1.amazonaws.com/language-models.pdf)).
 - GPT3 **Language models are few-shot learners**. T Brown et.al. **ICLR**, **2020**, **Number of Citations: **24996, ([pdf](./Papers/Language_models_are_few-shot_learners.pdf))([link](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)).



## <a id="Review">Review</a>
 - **Natural language processing in medicine: A review**. Locke Saskia et.al. **Trends in Anaesthesia and Critical Care**, **2021-6**, **Number of Citations: **85, ([pdf](./Papers/Natural_language_processing_in_medicine_A_review.pdf))([link](http://dx.doi.org/10.1016/j.tacc.2021.02.007)).
 - **A Survey of Text Representation and Embedding Techniques in NLP**. Patil Rajvardhan et.al. **IEEE Access**, **2023**, **Number of Citations: **12, ([pdf](./Papers/A_Survey_of_Text_Representation_and_Embedding_Techniques_in_NLP.pdf))([link](http://dx.doi.org/10.1109/access.2023.3266377)).
 - **Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing**. Pengfei Liu et.al. **arxiv**, **2021**, **Number of Citations: **3169, ([pdf](./Papers/Pre-train,_Prompt,_and_Predict_A_Systematic_Survey_of_Prompting_Methods_in_Natural_Language_Processing.pdf))([link](http://arxiv.org/abs/2107.13586v1)).
 - **In-context Learning with Retrieved Demonstrations for Language Models: A Survey**. Man Luo et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/In-context_Learning_with_Retrieved_Demonstrations_for_Language_Models_A_Survey.pdf))([link](http://arxiv.org/abs/2401.11624v5)).
 - **Transformers in Healthcare: A Survey**. Subhash Nerella et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/Transformers_in_Healthcare_A_Survey.pdf))([link](http://arxiv.org/abs/2307.00067v1)).
 - **A Survey on RAG Meets LLMs: Towards Retrieval-Augmented Large Language Models**. Yujuan Ding et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/A_Survey_on_RAG_Meets_LLMs_Towards_Retrieval-Augmented_Large_Language_Models.pdf))([link](http://arxiv.org/abs/2405.06211v1)).
 - **Deep learning joint models for extracting entities and relations in biomedical: a survey and comparison**. Y Su et.al. **Briefings Bioinform.**, **2022**, **Number of Citations: **4, ([pdf](./Papers/Deep_learning_joint_models_for_extracting_entities_and_relations_in_biomedical_a_survey_and_comparison.pdf))([link](https://academic.oup.com/bib/article-abstract/23/6/bbac342/6686739)).
 - **A survey of the recent trends in deep learning for literature based discovery in the biomedical domain**. Cesario Eugenio et.al. **Neurocomputing**, **2024-2**, **Number of Citations: **4, ([pdf](./Papers//A_survey_of_the_recent_trends_in_deep_learning_for_literature_based_discovery_in_the_biomedical_domain.pdf))([link](http://dx.doi.org/10.1016/j.neucom.2023.127079)).
 - **Recent Advances in Natural Language Processing via Large Pre-Trained Language Models: A Survey**. Bonan Min et.al. **arxiv**, **2021**, **Number of Citations: **None, ([pdf](./Papers/Recent_Advances_in_Natural_Language_Processing_via_Large_Pre-Trained_Language_Models_A_Survey.pdf))([link](http://arxiv.org/abs/2111.01243v1)).
 - **Deep learning for temporal data representation in electronic health records: A systematic review of challenges and methodologies**. Xie Feng et.al. **Journal of Biomedical Informatics**, **2022-2**, **Number of Citations: **43, ([pdf](./Papers/Deep_learning_for_temporal_data_representation_in_electronic_health_records_A_systematic_review_of_challenges_and_methodologies.pdf))([link](http://dx.doi.org/10.1016/j.jbi.2021.103980)).
 - **Large Language Models in Healthcare: A Review**. Zou Shun et.al. **No journal**, **2023-10-27**, **Number of Citations: **0, ([pdf](./Papers/Large_Language_Models_in_Healthcare_A_Review.pdf))([link](http://dx.doi.org/10.1109/iscsic60498.2023.00038)).
 - **On the Opportunities and Risks of Foundation Models**. Rishi Bommasani et.al. **arxiv**, **2021**, **Number of Citations: **None, ([pdf](./Papers/On_the_Opportunities_and_Risks_of_Foundation_Models.pdf))([link](http://arxiv.org/abs/2108.07258v3)).
 - **A Survey of Large Language Models**. Wayne Xin Zhao et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/A_Survey_of_Large_Language_Models.pdf))([link](http://arxiv.org/abs/2303.18223v13)).
 - **Several categories of large language models (llms): A short survey**. S Pahune et.al. **CoRR**, **2023**, **Number of Citations: **35, ([pdf](./Papers/Several_categories_of_large_language_models_(llms)_A_short_survey.pdf))([link](https://arxiv.org/abs/2307.10188)).
 - **Decoding Rarity: Large Language Models in the Diagnosis of Rare Diseases**. Valentina Carbonari et.al. **arxiv**, **2025**, **Number of Citations: **None, ([pdf](./Papers/Decoding_Rarity_Large_Language_Models_in_the_Diagnosis_of_Rare_Diseases.pdf))([link](http://arxiv.org/abs/2505.17065v1)).


## <a id="Models">Models</a>
 - **Publicly Available Clinical BERT Embeddings**. Emily Alsentzer et.al. **arxiv**, **2019**, **Number of Citations: **None, ([pdf](./Papers/Publicly_Available_Clinical_BERT_Embeddings.pdf))([link](http://arxiv.org/abs/1904.03323v3)).
 - **PMC-LLaMA: Towards Building Open-source Language Models for Medicine**. Chaoyi Wu et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/PMC-LLaMA_Towards_Building_Open-source_Language_Models_for_Medicine.pdf))([link](http://arxiv.org/abs/2304.14454v3)).
 - **ADELIE: Aligning Large Language Models on Information Extraction**. Yunjia Qi et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/ADELIE_Aligning_Large_Language_Models_on_Information_Extraction.pdf))([link](http://arxiv.org/abs/2405.05008v1)).
 - **IEPile: Unearthing Large-Scale Schema-Based Information Extraction
  Corpus**. Honghao Gui et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/IEPile_Unearthing_Large-Scale_Schema-Based_Information_Extraction_Corpus.pdf))([link](http://arxiv.org/abs/2402.14710v3)).
 - **AlpaCare:Instruction-tuned Large Language Models for Medical Application**. Xinlu Zhang et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/AlpaCareInstruction-tuned_Large_Language_Models_for_Medical_Application.pdf))([link](http://arxiv.org/abs/2310.14558v2)). 
    - Propose creating a diverse, machine-generated medical IFT dataset, MedInstruct-52k, using GPT-4 and ChatGPT with a high-quality expert-curated seed set.
    - LLaMA-series models
 - **MEDITRON-70B: Scaling Medical Pretraining for Large Language Models**. Zeming Chen et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/MEDITRON-70B_Scaling_Medical_Pretraining_for_Large_Language_Models.pdf))([link](http://arxiv.org/abs/2311.16079v1)). 
     - Based on Llama2 


## <a id="Datasets">Datasets</a>
 - **Does bert learn as humans perceive? understanding linguistic styles through lexica**. SA Hayati et.al. **EMNLP (1)**, **2021**, **Number of Citations: **24, ([pdf](./Papers/Does_bert_learn_as_humans_perceive_understanding_linguistic_styles_through_lexica.pdf))([link](https://arxiv.org/abs/2109.02738)).
     - HUMMINGBIRD dataset
 - **BioRED: a rich biomedical relation extraction dataset**. Luo Ling et.al. **No journal**, **2022-7-19**, **Number of Citations: **30, ([pdf](./Papers/BioRED_a_rich_biomedical_relation_extraction_dataset.pdf))([link](http://dx.doi.org/10.1093/bib/bbac282)).
 - **BioInfer: a corpus for information extraction in the biomedical domain**. Pyysalo Sampo et.al. **BMC Bioinformatics**, **2007-2-9**, **Number of Citations: **229, ([pdf](./Papers/BioInfer_a_corpus_for_information_extraction_in_the_biomedical_domain.pdf))([link](http://dx.doi.org/10.1186/1471-2105-8-50)).
 - **ODD: A Benchmark Dataset for the Natural Language Processing based Opioid Related Aberrant Behavior Detection**. Sunjae Kwon et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/ODD_A_Benchmark_Dataset_for_the_Natural_Language_Processing_based_Opioid_Related_Aberrant_Behavior_Detection.pdf))([link](http://arxiv.org/abs/2307.02591v4)).
 - **PHEE: A Dataset for Pharmacovigilance Event Extraction from Text**. Zhaoyue Sun et.al. **arxiv**, **2022**, **Number of Citations: **None, ([pdf](./Papers/PHEE_A_Dataset_for_Pharmacovigilance_Event_Extraction_from_Text.pdf))([link](http://arxiv.org/abs/2210.12560v1)).
 - **PubMedQA: A Dataset for Biomedical Research Question Answering**. Qiao Jin et.al. **arxiv**, **2019**, **Number of Citations: **None, ([pdf](./Papers/PubMedQA_A_Dataset_for_Biomedical_Research_Question_Answering.pdf))([link](http://arxiv.org/abs/1909.06146v1)).
   - [https://pubmedqa.github.io/](https://pubmedqa.github.io/)


## <a id="Evaluation">Evaluation</a>
 - **Gemini Goes to Med School: Exploring the Capabilities of Multimodal Large Language Models on Medical Challenge Problems & Hallucinations**. Ankit Pal et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/Gemini_Goes_to_Med_School_Exploring_the_Capabilities_of_Multimodal_Large_Language_Models_on_Medical_Challenge_Problems_&_Hallucinations.pdf))([link](http://arxiv.org/abs/2402.07023v1)). 
 - **Assessing The Potential Of Mid-Sized Language Models For Clinical QA**. Elliot Bolton et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/Assessing_The_Potential_Of_Mid-Sized_Language_Models_For_Clinical_QA.pdf))([link](http://arxiv.org/abs/2404.15894v1)).
 - **MedEval: A Multi-Level, Multi-Task, and Multi-Domain Medical Benchmark for Language Model Evaluation**. Zexue He et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/MedEval_A_Multi-Level,_Multi-Task,_and_Multi-Domain_Medical_Benchmark_for_Language_Model_Evaluation.pdf))([link](http://arxiv.org/abs/2310.14088v3)).
 - **Reasoning Over Pre-training: Evaluating LLM Performance and Augmentation in Women's Health**. Imprialou, M. et.al. **medrxiv**, **2025**, **Number of Citations: **None, ([pdf](./Papers/Reasoning_Over_Pre_training_Evaluating_LLM_Performance_and_Augmentation_in_Womens_Health.pdf))([link](https://www.biorxiv.org/content/10.1101/2025.05.22.25328162)).


## <a id="prompt">Prompting</a>
 - **Chain-of-thought prompting elicits reasoning in large language models**. J Wei et.al. **NeurIPS**, **2022**, **Number of Citations: **14819, ([pdf](./Papers/Chain-of-thought_prompting_elicits_reasoning_in_large_language_models.pdf))([link](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html?ref=https://githubhelp.com)).
 - **Self-consistency improves chain of thought reasoning in language models**. X Wang et.al. **ICLR**, **2022**, **Number of Citations: **1711, ([pdf](./Papers/Self-consistency_improves_chain_of_thought_reasoning_in_language_models.pdf))([link](https://arxiv.org/abs/2203.11171)).
 - **Tree of thoughts: Deliberate problem solving with large language models**. S Yao et.al. **NeurIPS**, **2023**, **Number of Citations: **3051, ([pdf](./Papers/Tree_of_thoughts_Deliberate_problem_solving_with_large_language_models.pdf))([link](https://proceedings.neurips.cc/paper_files/paper/2023/hash/271db9922b8d1f4dd7aaef84ed5ac703-Abstract-Conference.html)).



## <a id="RAG">RAG</a>
 - **A survey on rag meeting llms: Towards retrieval-augmented large language models**. W Fan et.al. **KDD**, **2024**, **Number of Citations: **140, ([pdf](./Papers/A_survey_on_rag_meeting_llms_Towards_retrieval-augmented_large_language_models.pdf))([link](https://dl.acm.org/doi/abs/10.1145/3637528.3671470)).
 - **Improving large language model applications in biomedicine with retrieval-augmented generation: a systematic review, meta-analysis, and clinical development guidelines**. Liu Siru et.al. **No journal**, **2025-1-15**, **Number of Citations: **0, ([pdf](./Papers/Improving_large_language_model_applications_in_biomedicine_with_retrieval-augmented_generation_a_systematic_review,_meta-analysis,_and_clinical_development_guidelines.pdf))([link](https://doi.org/10.1093/jamia/ocaf008)).
 - **Enhancing Retrieval-Augmented Generation: A Study of Best Practices**. Siran Li et.al. **arxiv**, **2025**, **Number of Citations: **None, ([pdf](./Papers/Enhancing_Retrieval-Augmented_Generation_A_Study_of_Best_Practices.pdf))([link](http://arxiv.org/abs/2501.07391v1)).
 - **Retrieval-augmented generation for knowledge-intensive nlp tasks**. P Lewis et.al. **NeurIPS**, **2020**, **Number of Citations: **1980, ([pdf](./Papers/Retrieval-augmented_generation_for_knowledge-intensive_nlp_tasks.pdf))([link](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)).
 - **Benchmarking large language models in retrieval-augmented generation**. J Chen et.al. **AAAI**, **2024**, **Number of Citations: **42, ([pdf](./Papers/Benchmarking_large_language_models_in_retrieval-augmented_generation.pdf))([link](https://ojs.aaai.org/index.php/AAAI/article/view/29728)).
 - **BMRetriever: Tuning Large Language Models as Better Biomedical Text Retrievers**. Ran Xu et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/BMRetriever_Tuning_Large_Language_Models_as_Better_Biomedical_Text_Retrievers.pdf))([link](http://arxiv.org/abs/2404.18443v1)). 
   - [https://huggingface.co/BMRetriever](https://huggingface.co/BMRetriever)
 - **REALM: Retrieval-Augmented Language Model Pre-Training**. Kelvin Guu et.al. **arxiv**, **2020**, **Number of Citations: **None, ([pdf](./Papers/REALM_Retrieval-Augmented_Language_Model_Pre-Training.pdf))([link](http://arxiv.org/abs/2002.08909v1)).
 - **STaRK: Benchmarking LLM Retrieval on Textual and Relational Knowledge Bases**. Shirley Wu et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/STaRK_Benchmarking_LLM_Retrieval_on_Textual_and_Relational_Knowledge_Bases.pdf))([link](http://arxiv.org/abs/2404.13207v1)).
 - **UPRISE: Universal Prompt Retrieval for Improving Zero-Shot Evaluation**. Daixuan Cheng et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/UPRISE_Universal_Prompt_Retrieval_for_Improving_Zero-Shot_Evaluation.pdf))([link](http://arxiv.org/abs/2303.08518v4)).
 - **BiomedRAG: A Retrieval Augmented Large Language Model for Biomedicine**. Mingchen Li et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/BiomedRAG_A_Retrieval_Augmented_Large_Language_Model_for_Biomedicine.pdf))([link](http://arxiv.org/abs/2405.00465v3)).
 - **Biomedical knowledge graph-optimized prompt generation for large language models**. Soman Karthik et.al. **No journal**, **2024-9**, **Number of Citations: **0, ([pdf](./Papers/Biomedical_knowledge_graph-optimized_prompt_generation_for_large_language_models.pdf))([link](https://doi.org/10.1093/bioinformatics/btae560)).
 - **KRAGEN: a knowledge graph-enhanced RAG framework for biomedical problem solving using large language models**. Matsumoto Nicholas et.al. **No journal**, **2024-6**, **Number of Citations: **2, ([pdf](./Papers/KRAGEN_a_knowledge_graph-enhanced_RAG_framework_for_biomedical_problem_solving_using_large_language_models.pdf))([link](https://doi.org/10.1093/bioinformatics/btae353)).
 - **LLMs are not Zero-Shot Reasoners for Biomedical Information Extraction**. Aishik Nagar et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/LLMs_are_not_Zero-Shot_Reasoners_for_Biomedical_Information_Extraction.pdf))([link](http://arxiv.org/abs/2408.12249v1)).
 - **BioRAG: A RAG-LLM Framework for Biological Question Reasoning**. Chengrui Wang et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/BioRAG_A_RAG-LLM_Framework_for_Biological_Question_Reasoning.pdf))([link](http://arxiv.org/abs/2408.01107v2)).
 - **Hybridrag: Integrating knowledge graphs and vector retrieval augmented generation for efficient information extraction**. B Sarmah et.al. **ICAIF**, **2024**, **Number of Citations: **6, ([pdf](./Papers/Hybridrag_Integrating_knowledge_graphs_and_vector_retrieval_augmented_generation_for_efficient_information_extraction.pdf))([link](https://dl.acm.org/doi/abs/10.1145/3677052.3698671)).
 - **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**. Akari Asai et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/Self_RAG_Learning_to_Retrieve_Generate_and_Critique_through_Self_Reflection.pdf))([link](http://arxiv.org/abs/2310.11511v1)).
 - **FB-RAG: Improving RAG with Forward and Backward Lookup**. Kushal Chawla et.al. **arxiv**, **2025**, **Number of Citations: **None, ([pdf](./Papers/FB-RAG_Improving_RAG_with_Forward_and_Backward_Lookup.pdf))([link](http://arxiv.org/abs/2505.17206v1)).
 - **The Role of Diversity in In-Context Learning for Large Language Models**. Wenyang Xiao et.al. **arxiv**, **2025**, **Number of Citations: **None, ([pdf](./Papers/The_Role_of_Diversity_in_In-Context_Learning_for_Large_Language_Models.pdf))([link](http://arxiv.org/abs/2505.19426v1)).

### Retrievers
 - **Unsupervised Dense Information Retrieval with Contrastive Learning**. Gautier Izacard et.al. **arxiv**, **2021**, **Number of Citations: **None, ([pdf](./Papers/Unsupervised_Dense_Information_Retrieval_with_Contrastive_Learning.pdf))([link](http://arxiv.org/abs/2112.09118v4)).
   - Contriever [https://huggingface.co/facebook/contriever](https://huggingface.co/facebook/contriever)
 - **How to Train Your DRAGON: Diverse Augmentation Towards Generalizable Dense Retrieval**. Sheng-Chieh Lin et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/How_to_Train_Your_DRAGON_Diverse_Augmentation_Towards_Generalizable_Dense_Retrieval.pdf))([link](http://arxiv.org/abs/2302.07452v1)).
   - Dragon [https://huggingface.co/facebook/dragon-plus-context-encoder](https://huggingface.co/facebook/dragon-plus-context-encoder)
 - **SciRepEval: A Multi-Format Benchmark for Scientific Document Representations**. Amanpreet Singh et.al. **arxiv**, **2022**, **Number of Citations: **None, ([pdf](./Papers/SciRepEval_A_Multi-Format_Benchmark_for_Scientific_Document_Representations.pdf))([link](http://arxiv.org/abs/2211.13308v4)).
   -  SPECTER2.0 [https://huggingface.co/allenai/specter2_base](https://huggingface.co/allenai/specter2_base)
 - **Pre-training Multi-task Contrastive Learning Models for Scientific Literature Understanding**. Yu Zhang et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/Pre-training_Multi-task_Contrastive_Learning_Models_for_Scientific_Literature_Understanding.pdf))([link](http://arxiv.org/abs/2305.14232v2)).
   - SciMult []()
 - **COCO-DR: Combating Distribution Shifts in Zero-Shot Dense Retrieval with Contrastive and Distributionally Robust Learning**. Yue Yu et.al. **arxiv**, **2022**, **Number of Citations: **None, ([pdf](./Papers/COCO-DR_Combating_Distribution_Shifts_in_Zero-Shot_Dense_Retrieval_with_Contrastive_and_Distributionally_Robust_Learning.pdf))([link](http://arxiv.org/abs/2210.15212v2)).
   - COCO-DR [https://huggingface.co/OpenMatch/cocodr-base-msmarco](https://huggingface.co/OpenMatch/cocodr-base-msmarco)
 - **SGPT: GPT Sentence Embeddings for Semantic Search**. Niklas Muennighoff et.al. **arxiv**, **2022**, **Number of Citations: **None, ([pdf](./Papers/SGPT_GPT_Sentence_Embeddings_for_Semantic_Search.pdf))([link](http://arxiv.org/abs/2202.08904v5)).
   - SGPT-125M [https://github.com/Muennighoff/sgpt?tab=readme-ov-file#use-sgpt-with-huggingface](https://github.com/Muennighoff/sgpt?tab=readme-ov-file#use-sgpt-with-huggingface)
 - **MedCPT: Contrastive Pre-trained Transformers with large-scale PubMed search logs for zero-shot biomedical information retrieval**. Jin Qiao et.al. **No journal**, **2023-11-1**, **Number of Citations: **10, ([pdf](./Papers/MedCPT_Contrastive_Pre-trained_Transformers_with_large-scale_PubMed_search_logs_for_zero-shot_biomedical_information_retrieval.pdf))([link](http://dx.doi.org/10.1093/bioinformatics/btad651)).
   - [https://huggingface.co/ncbi/MedCPT-Query-Encoder](https://huggingface.co/ncbi/MedCPT-Query-Encoder)
 - **Large Dual Encoders Are Generalizable Retrievers**. Jianmo Ni et.al. **arxiv**, **2021**, **Number of Citations: **None, ([pdf](./Papers/Large_Dual_Encoders_Are_Generalizable_Retrievers.pdf))([link](http://arxiv.org/abs/2112.07899v1)). 
   - GTR-L [https://huggingface.co/sentence-transformers/gtr-t5-large](https://huggingface.co/sentence-transformers/gtr-t5-large)
 - **One Embedder, Any Task: Instruction-Finetuned Text Embeddings**. Hongjin Su et.al. **arxiv**, **2022**, **Number of Citations: **None, ([pdf](./Papers/One_Embedder,_Any_Task_Instruction-Finetuned_Text_Embeddings.pdf))([link](http://arxiv.org/abs/2212.09741v3)).
   - InstructOR-L [https://huggingface.co/hkunlp/instructor-large](https://huggingface.co/hkunlp/instructor-large)
 - **Text Embeddings by Weakly-Supervised Contrastive Pre-training**. Liang Wang et.al. **arxiv**, **2022**, **Number of Citations: **None, ([pdf](./Papers/Text_Embeddings_by_Weakly-Supervised_Contrastive_Pre-training.pdf))([link](http://arxiv.org/abs/2212.03533v2)).
   - E5-Large-v2 [https://huggingface.co/intfloat/e5-large-v2](https://huggingface.co/intfloat/e5-large-v2)
 - **C-Pack: Packaged Resources To Advance General Chinese Embedding**. Shitao Xiao et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/C-Pack_Packaged_Resources_To_Advance_General_Chinese_Embedding.pdf))([link](http://arxiv.org/abs/2309.07597v4)).
   - BGE-Large [https://huggingface.co/BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)
 - **BMRetriever: Tuning Large Language Models as Better Biomedical Text Retrievers**. Ran Xu et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/BMRetriever_Tuning_Large_Language_Models_as_Better_Biomedical_Text_Retrievers.pdf))([link](http://arxiv.org/abs/2404.18443v1)). 
   - BMRETRIEVER-410M [https://huggingface.co/BMRetriever/BMRetriever-410M](https://huggingface.co/BMRetriever/BMRetriever-410M)

## Dietary supplement
 - **Deep learning approaches for extracting adverse events and indications of dietary supplements from clinical text**. Fan Yadan et.al. **No journal**, **2020-11-5**, **Number of Citations: **11, ([pdf](./Papers/Deep_learning_approaches_for_extracting_adverse_events_and_indications_of_dietary_supplements_from_clinical_text.pdf))([link](http://dx.doi.org/10.1093/jamia/ocaa218)).
 - **Identification of Dietary Supplement Use from Electronic Health Records Using Transformer-based Language Models**. Zhou Sicheng et.al. **No journal**, **2021-8**, **Number of Citations: **1, ([pdf](./Papers//Identification_of_Dietary_Supplement_Use_from_Electronic_Health_Records_Using_Transformer-based_Language_Models.pdf))([link](http://dx.doi.org/10.1109/ichi52183.2021.00096)).
 - **Ensemble BERT for classifying medication-mentioning tweets**. H Dang et.al. **SMM4H@COLING**, **2020**, **Number of Citations: **27, ([pdf](./Papers/Ensemble_BERT_for_classifying_medication-mentioning_tweets.pdf))([link](https://aclanthology.org/2020.smm4h-1.5/)).
 - **A conversational agent system for dietary supplements use**. E Singh et.al. **BMC Medical Informatics Decis. Mak.**, **2022**, **Number of Citations: **6, ([pdf](./Papers/A_conversational_agent_system_for_dietary_supplements_use.pdf))([link](https://link.springer.com/article/10.1186/s12911-022-01888-5)).
 - **Discovering novel drug-supplement interactions using a dietary supplements knowledge graph generated from the biomedical literature**. Dalton Schutte et.al. **arxiv**, **2021**, **Number of Citations: **None, ([pdf](./Papers/Discovering_novel_drug-supplement_interactions_using_a_dietary_supplements_knowledge_graph_generated_from_the_biomedical_literature.pdf))([link](http://arxiv.org/abs/2106.12741v1)).
 - **IUP-BERT: Identification of Umami Peptides Based on BERT Features**. Jiang Liangzhen et.al. **Foods**, **2022-11-21**, **Number of Citations: **13, ([pdf](./Papers/IUP-BERT_Identification_of_Umami_Peptides_Based_on_BERT_Features.pdf))([link](http://dx.doi.org/10.3390/foods11223742)).
 - **BERT-ER: Query-specific BERT Entity Representations for Entity Ranking.**. Shubham Chatterjee et.al. **SIGIR**, **2022**, **Number of Citations: **None, ([pdf](./Papers/BERT-ER_Query-specific_BERT_Entity_Representations_for_Entity_Ranking.pdf))([link](https://doi.org/10.1145/3477495.3531944)).
 - **Using natural language processing methods to classify use status of dietary supplements in clinical notes**. Fan Yadan et.al. **BMC Med Inform Decis Mak**, **2018-7**, **Number of Citations: **13, ([pdf](./Papers/Using_natural_language_processing_methods_to_classify_use_status_of_dietary_supplements_in_clinical_notes.pdf))([link](http://dx.doi.org/10.1186/s12911-018-0626-6)).


## <a id="grouppapers">Papers in our Group</a>
 - **Identifying Cardiomegaly in ChestX-ray8 Using Transfer Learning.**. Sicheng Zhou et.al. **MedInfo**, **2019**, **Number of Citations: **36, ([pdf](./Papers//Identifying_Cardiomegaly_in_ChestX-ray8_Using_Transfer_Learning.pdf.pdf))([link](https://doi.org/10.3233/SHTI190268)).
 - **Toward safer health care: a review strategy of FDA medical device adverse event database to identify and categorize health information technology related events**. Kang Hong et.al. **No journal**, **2018-10-12**, **Number of Citations: **6, ([pdf](./Papers/Toward_safer_health_care_a_review_strategy_of_FDA_medical_device_adverse_event_database_to_identify_and_categorize_health_information_technology_related_events.pdf))([link](http://dx.doi.org/10.1093/jamiaopen/ooy042)).
 - **Analysis of Twitter to Identify Topics Related to Eating Disorder Symptoms**. Zhou Sicheng et.al. **No journal**, **2019-6**, **Number of Citations: **10, ([pdf](./Papers/Analysis_of_Twitter_to_Identify_Topics_Related_to_Eating_Disorder_Symptoms.pdf))([link](http://dx.doi.org/10.1109/ichi.2019.8904863)).
 - **CancerBERT: a cancer domain-specific language model for extracting breast cancer phenotypes from electronic health records**. Zhou Sicheng et.al. **No journal**, **2022-3-25**, **Number of Citations: **36, ([pdf](./Papers/CancerBERT_a_cancer_domain-specific_language_model_for_extracting_breast_cancer_phenotypes_from_electronic_health_records.pdf))([link](http://dx.doi.org/10.1093/jamia/ocac040)).
 - **LEAP: LLM instruction-example adaptive prompting framework for biomedical relation extraction**. Zhou Huixue et.al. **No journal**, **2024-6-21**, **Number of Citations: **1, ([pdf](./Papers/LEAP_LLM_instruction-example_adaptive_prompting_framework_for_biomedical_relation_extraction.pdf))([link](http://dx.doi.org/10.1093/jamia/ocae147)).
 - **Repurposing Drugs for Alzheimer's Diseases through Link Prediction on Biomedical Literature**. Xiao Yongkang et.al. **No journal**, **2023-6-26**, **Number of Citations: **0, ([pdf](./Papers/Repurposing_Drugs_for_Alzheimers_Diseases_through_Link_Prediction_on_Biomedical_Literature.pdf))([link](http://dx.doi.org/10.1109/ichi57859.2023.00137)).
 - **Repurposing non-pharmacological interventions for Alzheimer's disease through link prediction on biomedical literature**. Xiao Yongkang et.al. **Sci Rep**, **2024-4-15**, **Number of Citations: **0, ([pdf](./Papers/Repurposing_non-pharmacological_interventions_for_Alzheimer's_disease_through_link_prediction_on_biomedical_literature.pdf))([link](http://dx.doi.org/10.1038/s41598-024-58604-8)). **Not Correct, Check it. Maybe mannual update & download is needed.**
 - **RT: a Retrieving and Chain-of-Thought framework for few-shot medical named entity recognition**. Li Mingchen et.al. **No journal**, **2024-5-6**, **Number of Citations: **0, ([pdf](./Papers/RT_a_Retrieving_and_Chain-of-Thought_framework_for_few-shot_medical_named_entity_recognition.pdf))([link](http://dx.doi.org/10.1093/jamia/ocae095)).
 - **Complementary and Integrative Health Information in the literature: its lexicon and named entity recognition**. Zhou Huixue et.al. **No journal**, **2023-11-10**, **Number of Citations: **5, ([pdf](./Papers/Complementary_and_Integrative_Health_Information_in_the_literature_its_lexicon_and_named_entity_recognition.pdf))([link](http://dx.doi.org/10.1093/jamia/ocad216)).
 - **Deep learning models in detection of dietary supplement adverse event signals from Twitter**. Wang Yefeng et.al. **No journal**, **2021-10-1**, **Number of Citations: **7, ([pdf](./Papers/Deep_learning_models_in_detection_of_dietary_supplement_adverse_event_signals_from_Twitter.pdf))([link](http://dx.doi.org/10.1093/jamiaopen/ooab081)).
 - **RAMIE: retrieval-augmented multi-task information extraction with large language models on dietary supplements**. Zhan Zaifu et.al. **No journal**, **2025-1-11**, **Number of Citations: **0, ([pdf](./Papers/RAMIE_retrieval-augmented_multi-task_information_extraction_with_large_language_models_on_dietary_supplements.pdf))([link](https://doi.org/10.1093/jamia/ocaf002)).
 - **CancerLLM: A Large Language Model in Cancer Domain**. Mingchen Li et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/CancerLLM_A_Large_Language_Model_in_Cancer_Domain.pdf))([link](http://arxiv.org/abs/2406.10459v2)).
 - **Multi-modality risk prediction of cardiovascular diseases for breast cancer cohort in the <i>All of Us</i> Research Program**. Yang Han et.al. **No journal**, **2024-7-26**, **Number of Citations: **0, ([pdf](./Papers/Multi-modality_risk_prediction_of_cardiovascular_diseases_for_breast_cancer_cohort_in_the_iAll_of_Usi_Research_Program.pdf))([link](https://doi.org/10.1093/jamia/ocae199)).
 - **Benchmarking large language models for biomedical natural language processing applications and recommendations**. Chen Qingyu et.al. **Nat Commun**, **2025-4-6**, **Number of Citations: **2, ([pdf](./Papers/Benchmarking_large_language_models_for_biomedical_natural_language_processing_applications_and_recommendations.pdf))([link](https://doi.org/10.1038/s41467-025-56989-2)).
 - **Large language models for disease diagnosis: a scoping review**. Zhou Shuang et.al. **npj Artif. Intell.**, **2025-6-9**, **Number of Citations: **25, ([pdf](./Papers/Large_language_models_for_disease_diagnosis_a_scoping_review.pdf))([link](https://doi.org/10.1038/s44387-025-00011-z)).

## <a id="fast_inference">Fast inference, Early exiting</a>
 - **A Survey on Model Compression and Acceleration for Pretrained Language Models**. Xu Canwen et.al. **AAAI**, **2023-6-26**, **Number of Citations: **11, ([pdf](./Papers/A_Survey_on_Model_Compression_and_Acceleration_for_Pretrained_Language_Models.pdf))([link](http://dx.doi.org/10.1609/aaai.v37i9.26255)).
 - **A Survey on Efficient Inference for Large Language Models**. Zixuan Zhou et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/A_Survey_on_Efficient_Inference_for_Large_Language_Models.pdf))([link](http://arxiv.org/abs/2404.14294v3)).
 - **Model Compression and Efficient Inference for Large Language Models: A Survey**. Wenxiao Wang et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/Model_Compression_and_Efficient_Inference_for_Large_Language_Models_A_Survey.pdf))([link](http://arxiv.org/abs/2402.09748v1)).

 - **BERT Loses Patience: Fast and Robust Inference with Early Exit**. Wangchunshu Zhou et.al. **arxiv**, **2020**, **Number of Citations: **None, ([pdf](./Papers/BERT_Loses_Patience_Fast_and_Robust_Inference_with_Early_Exit.pdf))([link](http://arxiv.org/abs/2006.04152v3)).
    - Code: [https://github.com/JetRunner/PABEE](https://github.com/JetRunner/PABEE)
 - **F-PABEE: Flexible-Patience-Based Early Exiting For Single-Label and Multi-Label Text Classification Tasks**. Gao Xiangxiang et.al. **No journal**, **2023-6-4**, **Number of Citations: **2, ([pdf](./Papers/F-PABEE_Flexible-Patience-Based_Early_Exiting_For_Single-Label_and_Multi-Label_Text_Classification_Tasks.pdf))([link](http://dx.doi.org/10.1109/icassp49357.2023.10095864)).
 - **Adaptive Inference through Early-Exit Networks**. Laskaridis Stefanos et.al. **No journal**, **2021-6-24**, **Number of Citations: **36, ([pdf](./Papers/Adaptive_Inference_through_Early-Exit_Networks.pdf))([link](http://dx.doi.org/10.1145/3469116.3470012)).
 - **PCEE-BERT: accelerating BERT inference via patient and confident early exiting**. Z Zhang et.al. **NAACL-HLT (Findings)**, **2022**, **Number of Citations: **23, ([pdf](./Papers/PCEE-BERT_accelerating_BERT_inference_via_patient_and_confident_early_exiting.pdf))([link](https://aclanthology.org/2022.findings-naacl.25/)).
 - **BERxiT: Early exiting for BERT with better fine-tuning and extension to regression**. J Xin et.al. **EACL**, **2021**, **Number of Citations: **98, ([pdf](./Papers/BERxiT_Early_exiting_for_BERT_with_better_fine-tuning_and_extension_to_regression.pdf))([link](https://aclanthology.org/2021.eacl-main.8/)).
 - **BranchySplit: Runtime-Adaptable Partitioning and Early Exits for Accelerated Edge Inference**. Berg Oscar Artur Bernd et.al. **IEEE Access**, **2026**, **Number of Citations: **0, ([pdf](Papers/BranchySplit_Runtime-Adaptable_Partitioning_and_Early_Exits_for_Accelerated_Edge_Inference.pdf))([link](https://doi.org/10.1109/access.2026.3651845)).
 - **Dataset Pruning Using Early Exit Networks**. Görmez Alperen et.al. **ACM Trans. Intell. Syst. Technol.**, **2026-2-19**, **Number of Citations: **0, ([pdf](Papers/Dataset_Pruning_Using_Early_Exit_Networks.pdf))([link](https://doi.org/10.1145/3785502)).
 - **ADEPT: Adaptive Dynamic Early-Exit Process for Transformers**. Sangmin Yoo et.al. **arxiv**, **2026**, **Number of Citations: **None, ([pdf](Papers/ADEPT_Adaptive_Dynamic_Early-Exit_Process_for_Transformers.pdf))([link](https://arxiv.org/abs/2601.03700v1)).
 - **Early-Exit and Instant Confidence Translation Quality Estimation**. ([pdf](./Papers//your_pdf_name.pdf???)).
 - **DART: Input-Difficulty-AwaRe Adaptive Threshold for Early-Exit DNNs**. Parth Patne et.al. **arxiv**, **2026**, **Number of Citations: **None, ([pdf](Papers/DART_Input-Difficulty-AwaRe_Adaptive_Threshold_for_Early-Exit_DNNs.pdf))([link](https://arxiv.org/abs/2603.12269v1)).
 - **The Diminishing Returns of Early-Exit Decoding in Modern LLMs**. Rui Wei et.al. **arxiv**, **2026**, **Number of Citations: **None, ([pdf](Papers/The_Diminishing_Returns_of_Early-Exit_Decoding_in_Modern_LLMs.pdf))([link](https://arxiv.org/abs/2603.23701v1)).
 - {{You need multiple exiting: Dynamic early exiting for accelerating unified vision language
model}}
 - **EnViT: Enhancing the Performance of Early-Exit Vision Transformers via Exit-Aware Structured Dropout-Enabled Self-Distillation**. Dong Yonghao et.al. **AAAI**, **2026-3-14**, **Number of Citations: **0, ([pdf](Papers/EnViT_Enhancing_the_Performance_of_Early-Exit_Vision_Transformers_via_Exit-Aware_Structured_Dropout-Enabled_Self-Distillation.pdf))([link](https://doi.org/10.1609/aaai.v40i25.39225)).

 - **PEER: Towards reliable and efficient inference via Patience-Based Early Exiting with Rejection**. Zhan Zaifu et.al. **Journal of Biomedical Informatics**, **2026-3**, **Number of Citations: **0, ([pdf](Papers/PEER_Towards_reliable_and_efficient_inference_via_Patience-Based_Early_Exiting_with_Rejection.pdf))([link](https://doi.org/10.1016/j.jbi.2026.104988)).

### biomedical
 - **Efficient Inference Of Image-Based Neural Network Models In Reconfigurable Systems With Pruning And Quantization.**. José Flich et.al. **ICIP**, **2022**, **Number of Citations: **None, ([pdf](./Papers/Efficient_Inference_Of_Image-Based_Neural_Network_Models_In_Reconfigurable_Systems_With_Pruning_And_Quantization.pdf))([link](https://doi.org/10.1109/ICIP46576.2022.9897752)).


## 2D Representation Info Extraction
 - **OneRel: Joint Entity and Relation Extraction with One Module in One Step**. Shang Yu-Ming et.al. **AAAI**, **2022-6-28**, **Number of Citations: **46, ([pdf](./Papers/OneRel_Joint_Entity_and_Relation_Extraction_with_One_Module_in_One_Step.pdf))([link](http://dx.doi.org/10.1609/aaai.v36i10.21379)).
 - **A Bi-consolidating Model for Joint Relational Triple Extraction**. Xiaocheng Luo et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/A_Bi-consolidating_Model_for_Joint_Relational_Triple_Extraction.pdf))([link](http://arxiv.org/abs/2404.03881v2)).
 - **A Two Dimensional Feature Engineering Method for Relation Extraction**. Hao Wang et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/A_Two_Dimensional_Feature_Engineering_Method_for_Relation_Extraction.pdf))([link](http://arxiv.org/abs/2404.04959v1)).


## <a id="Uncategorized">Uncategorized</a>
 - **JMLR: Joint Medical LLM and Retrieval Training for Enhancing Reasoning and Professional Question Answering Capability**. Junda Wang et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/JMLR_Joint_Medical_LLM_and_Retrieval_Training_for_Enhancing_Reasoning_and_Professional_Question_Answering_Capability.pdf))([link](http://arxiv.org/abs/2402.17887v3)). 
 - **Almanac: Retrieval-Augmented Language Models for Clinical Medicine**. Cyril Zakka et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/Almanac_Retrieval-Augmented_Language_Models_for_Clinical_Medicine.pdf))([link](http://arxiv.org/abs/2303.01229v2)).
 - **RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation**. Chao Jin et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/RAGCache_Efficient_Knowledge_Caching_for_Retrieval-Augmented_Generation.pdf))([link](http://arxiv.org/abs/2404.12457v2)). 
 - **Joint Biomedical Entity and Relation Extraction with Knowledge-Enhanced Collective Inference**. Tuan Lai et.al. **arxiv**, **2021**, **Number of Citations: **None, ([pdf](./Papers/Joint_Biomedical_Entity_and_Relation_Extraction_with_Knowledge-Enhanced_Collective_Inference.pdf))([link](http://arxiv.org/abs/2105.13456v2)).
 - **In-context Learning with Retrieved Demonstrations for Language Models: A Survey**. Man Luo et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/In-context_Learning_with_Retrieved_Demonstrations_for_Language_Models_A_Survey.pdf))([link](http://arxiv.org/abs/2401.11624v5)).
 - **Document-level biomedical relation extraction based on multi-dimensional fusion information and multi-granularity logical reasoning**. L Li et.al. **COLING**, **2022**, **Number of Citations: **7, ([pdf](./Papers/Document-level_biomedical_relation_extraction_based_on_multi-dimensional_fusion_information_and_multi-granularity_logical_reasoning.pdf))([link](https://aclanthology.org/2022.coling-1.183/)).
 - **Branch-Train-Merge: Embarrassingly Parallel Training of Expert Language Models**. Margaret Li et.al. **arxiv**, **2022**, **Number of Citations: **None, ([pdf](./Papers/Branch-Train-Merge_Embarrassingly_Parallel_Training_of_Expert_Language_Models.pdf))([link](http://arxiv.org/abs/2208.03306v1)).
 - **CRAFT: Customizing LLMs by Creating and Retrieving from Specialized Toolsets**. Lifan Yuan et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/CRAFT_Customizing_LLMs_by_Creating_and_Retrieving_from_Specialized_Toolsets.pdf))([link](http://arxiv.org/abs/2309.17428v2)).
 - **TrustLLM: Trustworthiness in Large Language Models**. Lichao Sun et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/TrustLLM_Trustworthiness_in_Large_Language_Models.pdf))([link](http://arxiv.org/abs/2401.05561v4)).
 - **GoLLIE: Annotation Guidelines improve Zero-Shot Information-Extraction**. Oscar Sainz et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/GoLLIE_Annotation_Guidelines_improve_Zero-Shot_Information-Extraction.pdf))([link](http://arxiv.org/abs/2310.03668v5)).
 - **Multi-task learning for natural language processing in the 2020s: Where are we going?**. Worsham Joseph et.al. **Pattern Recognition Letters**, **2020-8**, **Number of Citations: **43, ([pdf](./Papers/Multi-task_learning_for_natural_language_processing_in_the_2020s_Where_are_we_going.pdf))([link](http://dx.doi.org/10.1016/j.patrec.2020.05.031)).
 - **A Primer on Large Language Models and their Limitations**. Sandra Johnson et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/A_Primer_on_Large_Language_Models_and_their_Limitations.pdf))([link](http://arxiv.org/abs/2412.04503v1)).
 - **A surprisal oracle for when every layer counts**. Xudong Hong et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/A_surprisal_oracle_for_when_every_layer_counts.pdf))([link](http://arxiv.org/abs/2412.03098v1)).


## <a id="Instruction_Tuning">Instruction Tuning</a>
 - **Instruction Tuning for Large Language Models: A Survey**. Shengyu Zhang et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/Instruction_Tuning_for_Large_Language_Models_A_Survey.pdf))([link](http://arxiv.org/abs/2308.10792v5)).



## <a id="MTL">Multi-task Learning</a>
### Review
 - **Multi-task learning in natural language processing: An overview**. S Chen et.al. **CoRR**, **2021**, **Number of Citations: **60, ([pdf](./Papers/Multi-task_learning_in_natural_language_processing_An_overview.pdf))([link](https://dl.acm.org/doi/abs/10.1145/3663363)).
   - MTL is especially meaningful for low-resource tasks and languages whose labeled dataset is sometimes too small to sufficiently train a model
   - **Parallel architecture**: shares the bulk of the model among multiple tasks while each task has its own task-specific output layer.
   ![MTL_Parallel_Architectures](./Figures/MTL-Parallel_Architectures.png "Parallel Architectures")
   - **Hierarchical architecture**: hierarchically combine features from different tasks, take the output of one task as the input of another task, or explicitly model the interaction between tasks.
   ![MTL_Hierarchical_architectures](./Figures/MTL-hierarchical_architectures.png "Hierarchical_architectures")
   - **Modular architecture**: decomposes the whole model into shared components and task-specific components that learn task-invariant and task-specific features, respectively
   - **Generative adversarial architecture**
 - **Multi-task learning to improve natural language understanding**. Stefan Constantin et.al. **arxiv**, **2018**, **Number of Citations: **None, ([pdf](./Papers/Multi-task_learning_to_improve_natural_language_understanding.pdf))([link](http://arxiv.org/abs/1812.06876v2)).
 - **A Survey of Multi-task Learning in Natural Language Processing: Regarding Task Relatedness and Training Methods**. Zhihan Zhang et.al. **arxiv**, **2022**, **Number of Citations: **None, ([pdf](./Papers/A_Survey_of_Multi-task_Learning_in_Natural_Language_Processing_Regarding_Task_Relatedness_and_Training_Methods.pdf))([link](http://arxiv.org/abs/2204.03508v2)).
 - **An Overview of Multi-Task Learning in Deep Neural Networks**. Sebastian Ruder et.al. **arxiv**, **2017**, **Number of Citations: **None, ([pdf](./Papers/An_Overview_of_Multi-Task_Learning_in_Deep_Neural_Networks.pdf))([link](http://arxiv.org/abs/1706.05098v1)).
 - **Learning with Whom to Share in Multi-task Feature Learning.**. Zhuoliang Kang et.al. **ICML**, **2011**, **Number of Citations: **None, ([pdf](./Papers/Learning_with_Whom_to_Share_in_Multi-task_Feature_Learning.pdf))([link](https://icml.cc/2011/papers/344_icmlpaper.pdf)).
 - **Let the Model Decide its Curriculum for Multitask Learning**. Neeraj Varshney et.al. **arxiv**, **2022**, **Number of Citations: **None, ([pdf](./Papers/Let_the_Model_Decide_its_Curriculum_for_Multitask_Learning.pdf))([link](http://arxiv.org/abs/2205.09898v2)).
 - **Multi-Task Learning with Deep Neural Networks: A Survey**. Michael Crawshaw et.al. **arxiv**, **2020**, **Number of Citations: **None, ([pdf](./Papers/Multi-Task_Learning_with_Deep_Neural_Networks_A_Survey.pdf))([link](http://arxiv.org/abs/2009.09796v1)).
   - design of multi-task neural network architectures
   - MTL optimization methods into six distinct groups: loss weighting, regularization, gradient modulation, task scheduling, multi-objective optimization, and knowledge distillation.
   - The goal of TRL is to learn an explicit representation of tasks or relationships b
   - etween tasks, such as clustering tasks into groups by similarity, and leveraging the learned task relationships to improve learning on the tasks at hand.
 - **A brief review on multi-task learning**. KH Thung et.al. **Multim. Tools Appl.**, **2018**, **Number of Citations: **225, ([pdf](./Papers/A_brief_review_on_multi-task_learning.pdf))([link](https://link.springer.com/article/10.1007/s11042-018-6463-x)).

### Methodology
 - **An Empirical Study of Multi-Task Learning on BERT for Biomedical Text Mining**. Yifan Peng et.al. **arxiv**, **2020**, **Number of Citations: **None, ([pdf](./Papers/An_Empirical_Study_of_Multi-Task_Learning_on_BERT_for_Biomedical_Text_Mining.pdf))([link](http://arxiv.org/abs/2005.02799v1)).
   - Multi-task model: BERT, Shared Layers and Task specific layers
   - Datasets: Biomedical BLUE benchmark
 - **Identifying beneficial task relations for multi-task learning in deep neural networks**. Joachim Bingel et.al. **arxiv**, **2017**, **Number of Citations: **None, ([pdf](./Papers/Identifying_beneficial_task_relations_for_multi-task_learning_in_deep_neural_networks.pdf))([link](http://arxiv.org/abs/1702.08303v1)).
   - Model: bi-directional LSTM
   - 10 tasks
   - showing improvements in 40 outof 90 cases
   - fit a logarithmic function to the loss curve values
   - there was little evidence that dataset balance is a reliable predictor
 - **Which tasks should be learned together in multi-task learning?**. T Standley et.al. **ICML**, **2020**, **Number of Citations: **517, ([pdf](./Papers/Which_tasks_should_be_learned_together_in_multi-task_learning.pdf))([link](https://proceedings.mlr.press/v119/standley20a.html)).
   - Computer Vision
   - Better accuarcy and less inference time
   - Pairwise multi-task relationships
 - **Intermediate-Task Transfer Learning with Pretrained Models for Natural Language Understanding: When and Why Does It Work?**. Yada Pruksachatkun et.al. **arxiv**, **2020**, **Number of Citations: **None, ([pdf](./Papers/Intermediate-Task_Transfer_Learning_with_Pretrained_Models_for_Natural_Language_Understanding_When_and_Why_Does_It_Work.pdf))([link](http://arxiv.org/abs/2005.00628v2)).
   - RoBERTa model
   - 110 intermediate-target task combinations and 25 probing tasks
   - We observe that intermediate tasks requiring high-level inference and reasoning abilities tend to work best.
   - target task performance is strongly correlated with higher-level abilities such as coreference resolution
 - **A Hierarchical Multi-Task Approach for Learning Embeddings from Semantic Tasks**. Sanh Victor et.al. **AAAI**, **2019-7-17**, **Number of Citations: **71, ([pdf](./Papers/A_Hierarchical_Multi-Task_Approach_for_Learning_Embeddings_from_Semantic_Tasks.pdf))([link](http://dx.doi.org/10.1609/aaai.v33i01.33016949)).
   - The hierarchical training supervision induces a set of shared semantic representations at lower layers of the model.
 - **Exploring and Predicting Transferability across NLP Tasks**. Tu Vu et.al. **arxiv**, **2020**, **Number of Citations: **None, ([pdf](./Papers/Exploring_and_Predicting_Transferability_across_NLP_Tasks.pdf))([link](http://arxiv.org/abs/2005.00770v2)).
   - Fisher information matrix of the feature extractor to calculate relation, but is it good?
 - **AutoSeM: Automatic Task Selection and Mixing in Multi-Task Learning**. Han Guo et.al. **arxiv**, **2019**, **Number of Citations: **None, ([pdf](./Papers/AutoSeM_Automatic_Task_Selection_and_Mixing_in_Multi-Task_Learning.pdf))([link](http://arxiv.org/abs/1904.04153v1)).
   - multi-armed bandit controller used for task selection, Gaussian Process controller used for automatic mixing ratio (MR) learning
 - **Comic MTL: optimized multi-task learning for comic book image analysis**. Nguyen Nhu-Van et.al. **IJDAR**, **2019-7-17**, **Number of Citations: **19, ([pdf](./Papers/Comic_MTL_optimized_multi-task_learning_for_comic_book_image_analysis.pdf))([link](http://dx.doi.org/10.1007/s10032-019-00330-3)).
 - **Efficiently Identifying Task Groupings for Multi-Task Learning.**. Chris Fifty et.al. **NeurIPS**, **2021**, **Number of Citations: **None, ([pdf](./Papers/Efficiently_Identifying_Task_Groupings_for_Multi-Task_Learning.pdf))([link](https://proceedings.neurips.cc/paper/2021/hash/e77910ebb93b511588557806310f78f1-Abstract.html)).
   - We propose to measure inter-task affinity by training all tasks together in a single multi-task network and quantifying the effect to which one task gradient update would affect another task loss.
   - it computes task groupings from only a single training run.
 - **Towards Understanding Multi-Task Learning (Generalization) of LLMs via Detecting and Exploring Task-Specific Neurons**. Yongqi Leng et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/Towards_Understanding_Multi-Task_Learning_(Generalization)_of_LLMs_via_Detecting_and_Exploring_Task-Specific_Neurons.pdf))([link](http://arxiv.org/abs/2407.06488v1)).
   - the detected neurons are highly correlated with the given task, which we term as task-specific neurons.
   -  we propose a neuron-level continuous fine-tuning method that only fine-tunes the current task-specific neurons during continuous learning
 - **A Unified Multi-Task Learning Framework for Joint Extraction of Entities and Relations**. Zhao Tianyang et.al. **AAAI**, **2021-5-18**, **Number of Citations: **6, ([pdf](./Papers/A_Unified_Multi-Task_Learning_Framework_for_Joint_Extraction_of_Entities_and_Relations.pdf))([link](http://dx.doi.org/10.1609/aaai.v35i16.17707)).
 - **Joint Extraction of Entities and Relations Based on a Novel Tagging Scheme**. Suncong Zheng et.al. **arxiv**, **2017**, **Number of Citations: **None, ([pdf](./Papers/Joint_Extraction_of_Entities_and_Relations_Based_on_a_Novel_Tagging_Scheme.pdf))([link](http://arxiv.org/abs/1706.05075v1)).
 - **Enhancing Relation Extraction via Adversarial Multi-task Learning**. H Qin et.al. **LREC**, **2022**, **Number of Citations: **3, ([pdf](./Papers/Enhancing_Relation_Extraction_via_Adversarial_Multi-task_Learning.pdf))([link](https://aclanthology.org/2022.lrec-1.666/)).
   - Extract relation between two given named entities (NEs) 
   - we propose a RE model with two training stages, where adversarial multi-task learning is applied to the first training stage to explicitly recover the given NEs so as to enhance the main relation extractor, which is trained alone in the second stage.
   - Datasets: ACE2005EN(ACE05) and SemEval 2010 Task 8 (SemEval)

### Models
 - **Multi-task learning for few-shot biomedical relation extraction**. Moscato Vincenzo et.al. **Artif Intell Rev**, **2023-4-19**, **Number of Citations: **8, ([pdf](./Papers/Multi-task_learning_for_few-shot_biomedical_relation_extraction.pdf))([link](http://dx.doi.org/10.1007/s10462-023-10484-6)).
   - Multi-Task Deep Neural Network (MT-DNN)
   - DDI-2013, ChemProt, and I2B2-2010 RE
   - Our framework consists of a transformer-based model with shared layers across the three RE tasks, and  separate classification heads for each dataset. 
 - **MT-clinical BERT: scaling clinical information extraction with multitask learning**. Mulyar Andriy et.al. **No journal**, **2021-8-1**, **Number of Citations: **19, ([pdf](./Papers/MT-clinical_BERT_scaling_clinical_information_extraction_with_multitask_learning.pdf))([link](http://dx.doi.org/10.1093/jamia/ocab126)).
   - slight but consistent performance degradation in MT Clinical BERT relative to sequential fine-tuning
   - These results intuitively suggest that learning a general clinical text representation capable of supporting multiple tasks has the downside of losing the ability to exploit dataset or clinical note-specific properties
 whencomparedto asingle, task-specific model.
 - **BioInstruct: instruction tuning of large language models for biomedical natural language processing**. Tran Hieu et.al. **No journal**, **2024-6-4**, **Number of Citations: **0, ([pdf](./Papers/BioInstruct_instruction_tuning_of_large_language_models_for_biomedical_natural_language_processing.pdf))([link](http://dx.doi.org/10.1093/jamia/ocae122)).
   - Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning
   - Tasks: question answering (QA), information extraction (IE), and text generation (GEN)
   - 25 000 instructions: created by prompting the GPT-4 language model with 3-seed samples randomly drawn from an 80 human curated instructions
 - **Multi-Task Deep Neural Networks for Natural Language Understanding**. Xiaodong Liu et.al. **arxiv**, **2019**, **Number of Citations: **None, ([pdf](./Papers/Multi-Task_Deep_Neural_Networks_for_Natural_Language_Understanding.pdf))([link](http://arxiv.org/abs/1901.11504v2)).
 
 ![MTL](./Figures/1722612405642.png)

 - **A Novel Cascade Binary Tagging Framework for Relational Triple Extraction**. Zhepei Wei et.al. **arxiv**, **2019**, **Number of Citations: **None, ([pdf](./Papers/A_Novel_Cascade_Binary_Tagging_Framework_for_Relational_Triple_Extraction.pdf))([link](http://arxiv.org/abs/1909.03227v4)).
   -  solving the overlapping triple problem where multiple relational triples in the same sentence share the same entities
   -  our new framework models relations as functions that map subjects to objects in a sentence, which naturally handles the overlapping problem
   -  datasets NYT and WebNLG
 - 

 - **DeepStruct: Pretraining of Language Models for Structure Prediction**. Chenguang Wang et.al. **arxiv**, **2022**, **Number of Citations: **None, ([pdf](./Papers/DeepStruct_Pretraining_of_Language_Models_for_Structure_Prediction.pdf))([link](http://arxiv.org/abs/2205.10475v2)).
 - **InstructUIE: Multi-task Instruction Tuning for Unified Information Extraction**. Xiao Wang et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/InstructUIE_Multi-task_Instruction_Tuning_for_Unified_Information_Extraction.pdf))([link](http://arxiv.org/abs/2304.08085v1)).
 - **Code4Struct: Code Generation for Few-Shot Event Structure Prediction**. Xingyao Wang et.al. **arxiv**, **2022**, **Number of Citations: **None, ([pdf](./Papers/Code4Struct_Code_Generation_for_Few-Shot_Event_Structure_Prediction.pdf))([link](http://arxiv.org/abs/2210.12810v2)).
 - **GoLLIE: Annotation Guidelines improve Zero-Shot Information-Extraction**. Oscar Sainz et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/GoLLIE_Annotation_Guidelines_improve_Zero-Shot_Information-Extraction.pdf))([link](http://arxiv.org/abs/2310.03668v5)).
 - **ADELIE: Aligning Large Language Models on Information Extraction**. Yunjia Qi et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/ADELIE_Aligning_Large_Language_Models_on_Information_Extraction.pdf))([link](http://arxiv.org/abs/2405.05008v1)).
 - {{}}
 

 - **Unified Pre-training for Program Understanding and Generation**. Wasi Uddin Ahmad et.al. **arxiv**, **2021**, **Number of Citations: **None, ([pdf](./Papers/Unified_Pre-training_for_Program_Understanding_and_Generation.pdf))([link](http://arxiv.org/abs/2103.06333v2)).

 - **Enhancing relation extraction using multi-task learning with SDP evidence**. H Wang et.al. **Inf. Sci.**, **2024**, **Number of Citations: **0, ([pdf](./Papers/Enhancing_relation_extraction_using_multi-task_learning_with_SDP_evidence.pdf))([link](https://www.sciencedirect.com/science/article/pii/S0020025524005231)).
 - **Multi-task and multi-view training for end-to-end relation extraction**. Zhang Junchi et.al. **Neurocomputing**, **2019-10**, **Number of Citations: **12, ([pdf](./Papers/Multi-task_and_multi-view_training_for_end-to-end_relation_extraction.pdf))([link](http://dx.doi.org/10.1016/j.neucom.2019.06.087)).
 - **A neural network multi-task learning approach to biomedical named entity recognition**. G Crichton et.al. **BMC Bioinform.**, **2017**, **Number of Citations: **257, ([pdf](./Papers/A_neural_network_multi-task_learning_approach_to_biomedical_named_entity_recognition.pdf))([link](https://link.springer.com/article/10.1186/s12859-017-1776-8)).

 - **Neural multi-task learning in drug design**. ([pdf](./Papers//your_pdf_name.pdf)).

 - **Enhancing Mental Health Condition Detection on Social Media through Multi-Task Learning**. Liu, J. et.al. **medrxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/Enhancing_Mental_Health_Condition_Detection_on_Social_Media_through_Multi-Task_Learning.pdf))([link](https://www.biorxiv.org/content/10.1101/2024.02.23.24303303)).
 - **CT Multi-Task Learning with a Large Image-Text (LIT) Model**. Niu, C. et.al. **biorxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/CT_Multi-Task_Learning_with_a_Large_Image-Text_(LIT)_Model.pdf))([link](https://www.biorxiv.org/content/10.1101/2023.04.06.535859)).





## <a id="Detection">Detection</a>
 - **AID: Adaptive Integration of Detectors for Safe AI with Language Models**聽[pdf](./Papers/AID_Adaptive_Integration_of_Detectors_for_Safe_AI_with_Language_Models.pdf)
 - **Hate speech detection on Twitter using transfer learning**. Ali Raza et.al. **Computer Speech &amp; Language**, **2022-7**, **Number of Citations: **52, ([pdf](./Papers//your_pdf_name.pdf))([link](http://dx.doi.org/10.1016/j.csl.2022.101365)).
 - **Toxicity Detection with Generative Prompt-based Inference**. Yau-Shian Wang et.al. **arxiv**, **2022**, **Number of Citations: **None, ([pdf](./Papers/Toxicity_Detection_with_Generative_Prompt-based_Inference.pdf))([link](http://arxiv.org/abs/2205.12390v1)).
 - **Learning from the worst: Dynamically generated datasets to improve online hate detection**. B Vidgen et.al. **arXiv preprint arXiv:2012.15761**, **2020**, **Number of Citations: **208, ([pdf](./Papers/Learning_from_the_worst_Dynamically_generated_datasets_to_improve_online_hate_detection.pdf))([link](https://arxiv.org/abs/2012.15761)).
 - **Hate speech detection: Challenges and solutions**. MacAvaney Sean et.al. **PLoS ONE**, **2019-8-20**, **Number of Citations: **261, ([pdf](./Papers/Hate_speech_detection_Challenges_and_solutions.pdf))([link](http://dx.doi.org/10.1371/journal.pone.0221152)).
   - Datasets: HatebaseTwitter, WaseemA, WaseemB, Stormfront, TRAC, HatEval, Kaggle, GermanTwitter
 - **A Literature Review of Textual Hate Speech Detection Methods and Datasets**. Alkomah Fatimah et.al. **Information**, **2022-5-26**, **Number of Citations: **43, ([pdf](./Papers/A_Literature_Review_of_Textual_Hate_Speech_Detection_Methods_and_Datasets.pdf))([link](http://dx.doi.org/10.3390/info13060273)).
 - **Hate speech detection: A comprehensive review of recent works**. Gandhi Ankita et.al. **Expert Systems**, **2024-2-25**, **Number of Citations: **4, ([pdf](./Papers/Hate_speech_detection_A_comprehe_sive_review_of_recent_works.pdf))([link](http://dx.doi.org/10.1111/exsy.13562)).
 - **A survey on hate speech detection and sentiment analysis using machine learning and deep learning models**. Subramanian Malliga et.al. **Alexandria Engineering Journal**, **2023-10**, **Number of Citations: **11, ([pdf](./Papers/A_survey_on_hate_speech_detection_and_sentiment_analysis_using_machine_learning_and_deep_learning_models.pdf))([link](http://dx.doi.org/10.1016/j.aej.2023.08.038)).
 - **Deep learning for hate speech detection: a comparative study**. JS Malik et.al. **CoRR**, **2024**, **Number of Citations: **57, ([pdf](./Papers/Deep_learning_for_hate_speech_detection_a_comparative_study.pdf))([link](https://link.springer.com/article/10.1007/s41060-024-00650-6)).
 - **Hate speech detection in social media: Techniques, recent trends, and future challenges**. Rawat Anchal et.al. **WIREs Computational Stats**, **2024-3**, **Number of Citations: **2, ([pdf](./Papers/Hate_speech_detection_in_social_media_Techniques_recent_trends_and_future.pdf))([link](http://dx.doi.org/10.1002/wics.1648)).
 - **System 2 Attention (is something you might need too)**. Jason Weston et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/System_2_Attention_(is_something_you_might_need_too).pdf))([link](http://arxiv.org/abs/2311.11829v1)).
 - **HateDay: Insights from a Global Hate Speech Dataset Representative of a Day on Twitter**. Manuel Tonneau et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/HateDay_Insights_from_a_Global_Hate_Speech_Dataset_Representative_of_a_Day_on_Twitter.pdf))([link](http://arxiv.org/abs/2411.15462v1)).
   - HATEDAY consists of twelve annotated rep resentative sets (N=20K each) randomly sampled from  all tweets posted on September 21, 2022 in eight languages (Arabic, English, French, German, Indonesian, Portuguese, Spanish, and Turkish) and originating from four countries where English is the main language on Twitter (United States, India, Nigeria, Kenya).
 - **Directions in abusive language training data, a systematic review: Garbage in, garbage out**. Vidgen Bertie et.al. **PLoS ONE**, **2020-12-28**, **Number of Citations: **88, ([pdf](./Papers//Directions_in_abusive_language_training_data,_a_systematic_review_Garbage_in,_garbage_out.pdf))([link](http://dx.doi.org/10.1371/journal.pone.0243300)).
   - [https://hatespeechdata.com/](https://hatespeechdata.com/)



## <a id="FL">Federated Learning</a>
 - **An in-depth evaluation of federated learning on biomedical natural language processing for information extraction**. L Peng et.al. **npj Digit. Medicine**, **2024**, **Number of Citations: **3, ([pdf](./Papers//An_in-depth_evaluation_of_federated_learning_on_biomedical_natural_language_processing_for_information_extraction.pdf))([link](https://www.nature.com/articles/s41746-024-01126-4)).
 - **Federated Learning: Challenges, Methods, and Future Directions**. Li Tian et.al. **IEEE Signal Process. Mag.**, **2020-5**, **Number of Citations: **2613, ([pdf](./Papers/Federated_Learning_Challenges,_Methods,_and_Future_Directions.pdf))([link](https://doi.org/10.1109/msp.2020.2975749)).
 - **Advances and Open Problems in Federated Learning**. Peter Kairouz et.al. **arxiv**, **2019**, **Number of Citations: **None, ([pdf](./Papers/Advances_and_Open_Problems_in_Federated_Learning.pdf))([link](http://arxiv.org/abs/1912.04977v3)).
 - **A Survey on Securing Federated Learning: Analysis of Applications, Attacks, Challenges, and Trends.**. H茅lio N. Cunha Neto et.al. **IEEE Access**, **2023**, **Number of Citations: **None, ([pdf](./Papers/A_Survey_on_Securing_Federated_Learning_Analysis_of_Applications,_Attacks,_Challenges,_and_Trends.pdf))([link](https://doi.org/10.1109/ACCESS.2023.3269980)).
 - **Federated Learning for Generalization, Robustness, Fairness: A Survey and Benchmark**. Wenke Huang et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/Federated_Learning_for_Generalization,_Robustness,_Fairness_A_Survey_and_Benchmark.pdf))([link](http://arxiv.org/abs/2311.06750v1)).
 - **Open Challenges and Opportunities in Federated Foundation Models Towards Biomedical Healthcare**. Xingyu Li et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/Open_Challenges_and_Opportunities_in_Federated_Foundation_Models_Towards_Biomedical_Healthcare.pdf))([link](http://arxiv.org/abs/2405.06784v1)).
 - **Recent methodological advances in federated learning for healthcare**. Zhang Fan et.al. **Patterns**, **2024-6**, **Number of Citations: **0, ([pdf](./Papers/Recent_methodological_advances_in_federated_learning_for_healthcare.pdf))([link](https://doi.org/10.1016/j.patter.2024.101006)).
   - Five components: 1) Local data processing 2) Local Optimisation 3)Communication 4)Aggregation 5) Redistribution


## <a id="KVCache">KV Cache</a>
 - [https://huggingface.co/docs/transformers/main/en/kv_cache](https://huggingface.co/docs/transformers/main/en/kv_cache)
 - **KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.**. Zirui Liu et.al. **ICML**, **2024**, **Number of Citations: **None, ([pdf](./Papers/KIVI_A_Tuning-Free_Asymmetric_2bit_Quantization_for_KV_Cache.pdf))([link](https://openreview.net/forum?id=L057s2Rq8O)).


## <a id="Diffusionmodels">Diffusion model</a>
 - **Diffusion models in bioinformatics and computational biology**. Guo Zhiye et.al. **Nat Rev Bioeng**, **2023-10-27**, **Number of Citations: **19, ([pdf](./Papers/Diffusion_models_in_bioinformatics_and_computational_biology.pdf))([link](https://doi.org/10.1038/s44222-023-00114-9)).
 - **Diffusion models in text generation: a survey**. Yi Qiuhua et.al. **No journal**, **2024-2-23**, **Number of Citations: **1, ([pdf](./Papers/Diffusion_models_in_text_generation_a_survey.pdf))([link](https://doi.org/10.7717/peerj-cs.1905)).


## <a id="PEFT">PEFT</a>
 - **Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey**. Zeyu Han et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/Parameter-Efficient_Fine-Tuning_for_Large_Models_A_Comprehensive_Survey.pdf))([link](http://arxiv.org/abs/2403.14608v7)).
 - **FLoRA: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations.**. Ziyao Wang et.al. **NeurIPS**, **2024**, **Number of Citations: **None, ([pdf](./Papers/FLoRA_Federated_Fine_Tuning_large_language_models_with_heterogeneous_low_rank_adaptations.pdf))([link](http://papers.nips.cc/paper_files/paper/2024/hash/28312c9491d60ed0c77f7fff4ad86dd1-Abstract-Conference.html)).


## Multimodal
 - **A scoping review on multimodal deep learning in biomedical images and texts**. Sun Zhaoyi et.al. **Journal of Biomedical Informatics**, **2023-10**, **Number of Citations: **5, ([pdf](./Papers/A_scoping_review_on_multimodal_deep_learning_in_biomedical_images_and_texts.pdf))([link](https://doi.org/10.1016/j.jbi.2023.104482)).
 - **BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs**. Sheng Zhang et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/BiomedCLIP_a_multimodal_biomedical_foundation_model_pretrained_from_fifteen_million_scientific_image-text_pairs.pdf))([link](http://arxiv.org/abs/2303.00915v3)).
 - **Diagnostic Captioning: A Survey**. John Pavlopoulos et.al. **arxiv**, **2021**, **Number of Citations: **None, ([pdf](./Papers/Diagnostic_Captioning_A_Survey.pdf))([link](http://arxiv.org/abs/2101.07299v1)).
 - **Medical Image Captioning via Generative Pretrained Transformers**. Alexander Selivanov et.al. **arxiv**, **2022**, **Number of Citations: **None, ([pdf](./Papers/Medical_Image_Captioning_via_Generative_Pretrained_Transformers.pdf))([link](http://arxiv.org/abs/2209.13983v1)).
 - **Sam-Guided Enhanced Fine-Grained Encoding with Mixed Semantic Learning for Medical Image Captioning**. Zhenyu Zhang et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/Sam-Guided_Enhanced_Fine-Grained_Encoding_with_Mixed_Semantic_Learning_for_Medical_Image_Captioning.pdf))([link](http://arxiv.org/abs/2311.01004v2)).
 - **Retrieval Augmented Chest X-Ray Report Generation using OpenAI GPT
  models**. Mercy Ranjit et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/Retrieval_Augmented_Chest_X-Ray_Report_Generation_using_OpenAI_GPT_models.pdf))([link](http://arxiv.org/abs/2305.03660v1)).
 - **LaB-RAG: Label Boosted Retrieval Augmented Generation for Radiology
  Report Generation**. Steven Song et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/LaB-RAG_Label_Boosted_Retrieval_Augmented_Generation_for_Radiology_Report_Generation.pdf))([link](http://arxiv.org/abs/2411.16523v1)).
 - **MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models**. Peng Xia et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf))([link](http://arxiv.org/abs/2410.13085v1)).



## Distil
 - **Distilling large language models for matching patients to clinical trials**. Nievas Mauro et.al. **No journal**, **2024-4-19**, **Number of Citations: **10, ([pdf](./Papers/Distilling_large_language_models_for_matching_patients_to_clinical_trials.pdf))([link](https://doi.org/10.1093/jamia/ocae073)).
 - **A Survey on Knowledge Distillation of Large Language Models**. Xiaohan Xu et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/A_Survey_on_Knowledge_Distillation_of_Large_Language_Models.pdf))([link](http://arxiv.org/abs/2402.13116v4)).
 - **Survey on Knowledge Distillation for Large Language Models: Methods,
  Evaluation, and Application**. Chuanpeng Yang et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/Survey_on_Knowledge_Distillation_for_Large_Language_Models_Methods,_Evaluation,_and_Application.pdf))([link](http://arxiv.org/abs/2407.01885v1)).
 - **A Survey on Symbolic Knowledge Distillation of Large Language Models**. Acharya Kamal et.al. **IEEE Trans. Artif. Intell.**, **2024-12**, **Number of Citations: **1, ([pdf](./Papers/A_Survey_on_Symbolic_Knowledge_Distillation_of_Large_Language_Models.pdf))([link](https://doi.org/10.1109/tai.2024.3428519)).
 - **A Survey on Model Compression for Large Language Models**. Xunyu Zhu et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/A_Survey_on_Model_Compression_for_Large_Language_Models.pdf))([link](http://arxiv.org/abs/2308.07633v4)).
 - **Distilling Large Language Models for Efficient Clinical Information Extraction**. Karthik S. Vedula et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/Distilling_Large_Language_Models_for_Efficient_Clinical_Information_Extraction.pdf))([link](http://arxiv.org/abs/2501.00031v1)).
 - **Distilling Large Language Models for Biomedical Knowledge Extraction: A Case Study on Adverse Drug Events**. Yu Gu et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/Distilling_Large_Language_Models_for_Biomedical_Knowledge_Extraction_A_Case_Study_on_Adverse_Drug_Events.pdf))([link](http://arxiv.org/abs/2307.06439v1)).
 - **Lion: Adversarial Distillation of Proprietary Large Language Models**. Yuxin Jiang et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/Lion_Adversarial_Distillation_of_Proprietary_Large_Language_Models.pdf))([link](http://arxiv.org/abs/2305.12870v2)).


## <a id="QA">QA</a>
 - **One LLM is not Enough: Harnessing the Power of Ensemble Learning for Medical Question Answering**. Yang, H. et.al. **medrxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/One_LLM_is_not_Enough_Harnessing_the_Power_of_Ensemble_Learning_for_Medical_Question_Answering.pdf))([link](https://www.biorxiv.org/content/10.1101/2023.12.21.23300380)).

 Here are the related datasets for LLM reasoning.
1. Medbullets: https://huggingface.co/datasets/LangAGI-Lab/medbullets,
2. MMLU-Pro: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro,
3. MedExQA: https://huggingface.co/datasets/bluesky333/MedExQA,
4. MedXpertQA: https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA,
5. Humanity’s Last Exam: https://github.com/centerforaisafety/hle,
6. MedQA-USMLE: https://huggingface.co/datasets/bigbio/med_qa,
7. PubMedQA: https://huggingface.co/datasets/bigbio/pubmed_qa,
8. MedMCQA: https://huggingface.co/datasets/lighteval/med_mcqa,
9. MMLU-Medicine: https://huggingface.co/datasets/cais/mmlu,
10. HEAD-QA: https://huggingface.co/datasets/dvilares/head_qa.

11. DiagnosisArena: https://github.com/SPIRAL-MED/DiagnosisArena
12 MedCaseReasoning: https://huggingface.co/datasets/zou-lab/MedCaseReasoning


## <a id="RAG+MLLM">RAG+MLLM</a>
 - **AlzheimerRAG: Multimodal Retrieval Augmented Generation for PubMed articles**. Aritra Kumar Lahiri et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/AlzheimerRAG_Multimodal_Retrieval_Augmented_Generation_for_PubMed_articles.pdf))([link](http://arxiv.org/abs/2412.16701v1)).
 - **RAMM: Retrieval-augmented Biomedical Visual Question Answering with Multi-modal Pre-training**. Zheng Yuan et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/RAMM_Retrieval-augmented_Biomedical_Visual_Question_Answering_with_Multi-modal_Pre-training.pdf))([link](http://arxiv.org/abs/2303.00534v1)).
 - **Retrieving Multimodal Information for Augmented Generation: A Survey**. Ruochen Zhao et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/Retrieving_Multimodal_Information_for_Augmented_Generation_A_Survey.pdf))([link](http://arxiv.org/abs/2303.10868v3)).
 - **Reducing Hallucinations of Medical Multimodal Large Language Models with Visual Retrieval-Augmented Generation**. Yun-Wei Chu et.al. **arxiv**, **2025**, **Number of Citations: **None, ([pdf](./Papers/Reducing_Hallucinations_of_Medical_Multimodal_Large_Language_Models_with_Visual_Retrieval-Augmented_Generation.pdf))([link](http://arxiv.org/abs/2502.15040v1)).
 - **MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models**. Peng Xia et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf))([link](http://arxiv.org/abs/2410.13085v2)).
 - **RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models**. Peng Xia et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/RULE_Reliable_Multimodal_RAG_for_Factuality_in_Medical_Vision_Language_Models.pdf))([link](http://arxiv.org/abs/2407.05131v2)).
 - **Beyond Text: Optimizing RAG with Multimodal Inputs for Industrial Applications**. Monica Riedler et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/Beyond_Text_Optimizing_RAG_with_Multimodal_Inputs_for_Industrial_Applications.pdf))([link](http://arxiv.org/abs/2410.21943v1)).
 - **MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models**. Peng Xia et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/MMed-RAG_Versatile_Multimodal_RAG_System_for_Medical_Vision_Language_Models.pdf))([link](http://arxiv.org/abs/2410.13085v2)).
 - **Benchmarking Retrieval-Augmented Generation for Medicine.**. Guangzhi Xiong et.al. **ACL (Findings)**, **2024**, **Number of Citations: **None, ([pdf](./Papers/Benchmarking_Retrieval-Augmented_Generation_for_Medicine.pdf))([link](https://doi.org/10.18653/v1/2024.findings-acl.372)).



## <a id="Cohort_study">Cohort study</a>
 - **Association of Preterm Singleton Birth With Fertility Treatment in the US**. Wang Ran et.al. **JAMA Netw Open**, **2022-2-8**, **Number of Citations: **10, ([pdf](./Papers/Association_of_Preterm_Singleton_Birth_With_Fertility_Treatment_in_the_US.pdf))([link](https://doi.org/10.1001/jamanetworkopen.2021.47782)).




## <a id="selective_classification">Selective classification</a>
 - **How to Fix a Broken Confidence Estimator: Evaluating Post-hoc Methods for Selective Classification with Deep Neural Networks.**. Lu��s Felipe P. Cattelan et.al. **UAI**, **2024**, **Number of Citations: **None, ([pdf](./Papers//your_pdf_name.pdf))([link](https://proceedings.mlr.press/v244/cattelan24a.html)).
 - **Towards Better Selective Classification**. Leo Feng et.al. **arxiv**, **2022**, **Number of Citations: **None, ([pdf](./Papers/Towards_Better_Selective_Classification.pdf))([link](http://arxiv.org/abs/2206.09034v4)).
 - {{}}

## <a id="model_merge">Model Merge</a>
 [Github: Awesome-Model-Merging-Methods-Theories-Applications](https://github.com/EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications?tab=readme-ov-file#weighted-based-merging-methods)
 - **Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities**. Enneng Yang et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/Model_Merging_in_LLMs_MLLMs_and_Beyond_Methods_Theories_Applications_and_Opportunities.pdf))([link](http://arxiv.org/abs/2408.07666v4)).
 - **Merging Models with Fisher-Weighted Averaging**. Michael Matena et.al. **arxiv**, **2021**, **Number of Citations: **None, ([pdf](./Papers/Merging_Models_with_Fisher-Weighted_Averaging.pdf))([link](http://arxiv.org/abs/2111.09832v2)).
 - **STAR: Spectral Truncation and Rescale for Model Merging**. Yu-Ang Lee et.al. **arxiv**, **2025**, **Number of Citations: **None, ([pdf](./Papers/STAR_Spectral_Truncation_and_Rescale_for_Model_Merging.pdf))([link](http://arxiv.org/abs/2502.10339v1)).
 - **MergeBench: A Benchmark for Merging Domain-Specialized LLMs**. Yifei He et.al. **arxiv**, **2025**, **Number of Citations: **None, ([pdf](./Papers/MergeBench_A_Benchmark_for_Merging_Domain-Specialized_LLMs.pdf))([link](http://arxiv.org/abs/2505.10833v2)).


## <a id="quantization">Quantization</a>
 - **Benchmarking large language models for biomedical natural language processing applications and recommendations**. Chen Qingyu et.al. **Nat Commun**, **2025-4-6**, **Number of Citations: **2, ([pdf](./Papers/Benchmarking_large_language_models_for_biomedical_natural_language_processing_applications_and_recommendations.pdf))([link](https://doi.org/10.1038/s41467-025-56989-2)).

| Task Category                    | Dataset Name                    | Description                                                        |
|----------------------------------|----------------------------------|--------------------------------------------------------------------|
| **Named Entity Recognition**     | BC5CDR-chemical                  | Chemical named entity recognition                                 |
|                                  | NCBI-disease                     | Disease named entity recognition                                  |
| **Relation Extraction**          | ChemProt                         | Chemical-protein relation extraction                              |
|                                  | DDI2013                          | Drug-drug interaction relation extraction                         |
| **Multi-label Classification**   | HoC (Hallmarks of Cancer)        | Multi-label classification for cancer-related literature          |
|                                  | LitCovid                         | Multi-label classification for COVID-19 literature                |
| **Question Answering**           | MedQA5-option                    | Multiple-choice medical QA                                        |
|                                  | PubMedQA                         | QA task based on PubMed biomedical articles                       |
| **Text Summarization**           | PubMed Text Summarization        | Summarization of biomedical research articles                     |
|                                  | MS² (Multi-Document Summarization) | Summarization from multiple scientific documents                 |
| **Text Simplification**          | Cochrane PLS                     | Plain Language Summaries for systematic reviews                   |
|                                  | PLOS Text Simplification         | Simplification of biomedical research articles from PLOS journals |


### Quantization methods
 - **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration**. Ji Lin et.al. **arxiv**, **2023**, **Number of Citations: **None, ([pdf](./Papers/AWQ_Activation-aware_Weight_Quantization_for_LLM_Compression_and_Acceleration.pdf))([link](http://arxiv.org/abs/2306.00978v5)).
 - **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale**. Tim Dettmers et.al. **arxiv**, **2022**, **Number of Citations: **None, ([pdf](./Papers/LLMint8()_8-bit_Matrix_Multiplication_for_Transformers_at_Scale.pdf))([link](http://arxiv.org/abs/2208.07339v2)).
 - **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers**. Elias Frantar et.al. **arxiv**, **2022**, **Number of Citations: **None, ([pdf](./Papers/GPTQ_Accurate_Post-Training_Quantization_for_Generative_Pre-trained_Transformers.pdf))([link](http://arxiv.org/abs/2210.17323v2)).
 - **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models**. Guangxuan Xiao et.al. **arxiv**, **2022**, **Number of Citations: **None, ([pdf](./Papers/SmoothQuant_Accurate_and_Efficient_Post-Training_Quantization_for_Large_Language_Models.pdf))([link](http://arxiv.org/abs/2211.10438v7)).
 - **Kivi: A tuning-free asymmetric 2bit quantization for kv cache**. Z Liu et.al. **ICML**, **2024**, **Number of Citations: **163, ([pdf](./Papers/Kivi_A_tuning-free_asymmetric_2bit_quantization_for_kv_cache.pdf))([link](https://arxiv.org/abs/2402.02750)).

### Evaluation
 - **Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation**. Yao Zhewei et.al. **AAAI**, **2024-3-24**, **Number of Citations: **5, ([pdf](./Papers/Exploring_Post-training_Quantization_in_LLMs_from_Comprehensive_Study_to_Low_Rank_Compensation.pdf))([link](https://doi.org/10.1609/aaai.v38i17.29908)).
 - **Do emergent abilities exist in quantized large language models: An empirical study**. P Liu et.al. **LREC/COLING**, **2023**, **Number of Citations: **33, ([pdf](./Papers/Do_emergent_abilities_exist_in_quantized_large_language_models_An_empirical_study.pdf))([link](https://arxiv.org/abs/2307.08072)).
 - **Evaluating Quantized Large Language Models**. Shiyao Li et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/Evaluating_Quantized_Large_Language_Models.pdf))([link](http://arxiv.org/abs/2402.18158v2)).
 - **Benchmarking Post-Training Quantization in LLMs: Comprehensive Taxonomy, Unified Evaluation, and Comparative Analysis**. Jiaqi Zhao et.al. **arxiv**, **2025**, **Number of Citations: **None, ([pdf](./Papers/Benchmarking_Post-Training_Quantization_in_LLMs_Comprehensive_Taxonomy_Unified_Evaluation_and_Comparative_Analysis.pdf))([link](http://arxiv.org/abs/2502.13178v4)).
 - **Evaluating the Generalization Ability of Quantized LLMs: Benchmark, Analysis, and Toolbox**. Yijun Liu et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/Evaluating_the_Generalization_Ability_of_Quantized_LLMs_Benchmark_Analysis_and_Toolbox.pdf))([link](http://arxiv.org/abs/2406.12928v1)).
 - **Quantized Large Language Models for Mental Health Applications: A Benchmark Study on Efficiency, Accuracy and Resource Allocation**. ([pdf](./Papers/Quantized_Large_Language_Models_for_Mental_Health_Applications_A_Benchmark_Study_on_Efficiency_Accuracy_and_Resource_Allocation.pdf))([link](https://search.proquest.com/openview/ac1c8e7459143637e9a17c1ebc404637/1?pq-origsite=gscholar&cbl=18750&diss=y)).

### Application
 - **The Rise of Small Language Models in Healthcare: A Comprehensive Survey**. Muskan Garg et.al. **arxiv**, **2025**, **Number of Citations: **None, ([pdf](./Papers/The_Rise_of_Small_Language_Models_in_Healthcare_A_Comprehensive_Survey.pdf))([link](http://arxiv.org/abs/2504.17119v2)).
 - **Efficient Biomedical Text Summarization With Quantized <scp>LLaMA</scp> 2: Enhancing Memory Usage and Inference on Low Powered Devices**. Kumar Sanjeev et.al. **Expert Systems**, **2024-10-27**, **Number of Citations: **2, ([pdf](./Papers/Efficient_Biomedical_Text_Summarization_With_Quantized_LLaMA_2_Enhancing_Memory_Usage_and_Inference_on_Low_Powered_Devices.pdf))([link](https://doi.org/10.1111/exsy.13760)).
 - **QM-ToT: A Medical Tree of Thoughts Reasoning Framework for Quantized Model**. Zongxian Yang et.al. **arxiv**, **2025**, **Number of Citations: **None, ([pdf](./Papers/QM-ToT_A_Medical_Tree_of_Thoughts_Reasoning_Framework_for_Quantized_Model.pdf))([link](http://arxiv.org/abs/2504.12334v1)).
 - **Privacy-Preserving SAM Quantization for Efficient Edge Intelligence in Healthcare**. Zhikai Li et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/Privacy-Preserving_SAM_Quantization_for_Efficient_Edge_Intelligence_in_Healthcare.pdf))([link](http://arxiv.org/abs/2410.01813v1)).
 - **Reasoning language models for more transparent prediction of suicide risk**. TH McCoy et.al. **BMJ Mental Health**, **2025**, **Number of Citations: **0, ([pdf](./Papers/Reasoning_language_models_for_more_transparent_prediction_of_suicide_risk.pdf))([link](https://mentalhealth.bmj.com/content/28/1/e301654)).
 - **Mental Healthcare Chatbot Based on Custom Diagnosis Documents Using a Quantized Large Language Model**. Kumar Ayush et.al. **No journal**, **2024-3-14**, **Number of Citations: **0, ([pdf](./Papers/Mental_Healthcare_Chatbot_Based_on_Custom_Diagnosis_Documents_Using_a_Quantized_Large_Language_Model.pdf))([link](https://doi.org/10.1109/icrito61523.2024.10522371)).
 - **MentalQLM: A lightweight large language model for mental healthcare based on instruction tuning and dual LoRA modules**. Shi, J. et.al. **medrxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/MentalQLM_A_lightweight_large_language_model_for_mental_healthcare_based_on_instruction_tuning_and_dual_LoRA_modules.pdf))([link](https://www.biorxiv.org/content/10.1101/2024.12.29.24319755)).


## <a id="complication">Complication + Diagnosis</a>
 - **Application of large language models in disease diagnosis and treatment**. X Yang et.al. **Chinese Medical ��**, **2025**, **Number of Citations: **7, ([pdf](./Papers/Application_of_large_language_models_in_disease_diagnosis_and_treatment.pdf))([link](https://mednexus.org/doi/abs/10.1097/CM9.0000000000003456)).
 - **Evaluation of the Potential Utility of an Artificial Intelligence Chatbot in Gastroesophageal Reflux Disease Management**. ([pdf](./Papers/Evaluation_of_the_Potential_Utility_of_an_Artificial_Intelligence_Chatbot_in_Gastroesophageal_Reflux_Disease_Management.pdf)).
 - **Enhancing Multi-Class Disease Classification: Neoplasms, Cardiovascular, Nervous System, and Digestive Disorders Using Advanced LLMs**. Ahmed Akib Jawad Karim et.al. **arxiv**, **2024**, **Number of Citations: **None, ([pdf](./Papers/Enhancing_Multi-Class_Disease_Classification_Neoplasms_Cardiovascular_Nervous_System_and_Digestive_Disorders_Using_Advanced_LLMs.pdf))([link](http://arxiv.org/abs/2411.12712v1)).

 - **MOVER: Medical Informatics Operating Room Vitals and Events Repository**. M Samad et.al. **nothing**, **2023**, **Number of Citations: **3, ([pdf](./Papers//your_pdf_name.pdf))([link](https://www.medrxiv.org/content/10.1101/2023.03.03.23286777.abstract)).
 - **Complications and morbidity following breast reconstruction �C a review of 16,063 cases from the 2005�C2010 NSQIP datasets**. Fischer John P. et.al. **Journal of Plastic Surgery and Hand Surgery**, **2013-7-18**, **Number of Citations: **97, ([pdf](./Papers/Complications_and_morbidity_following_breast_reconstruction_�C_a_review_of_16,063_cases_from_the_2005�C2010_NSQIP_datasets.pdf))([link](https://doi.org/10.3109/2000656x.2013.819003)).
 - **Benchmarking complications associated with esophagectomy**. DE Low et.al. **Annals of ��**, **2019**, **Number of Citations: **812, ([pdf](./Papers//your_pdf_name.pdf))([link](https://journals.lww.com/annalsofsurgery/fulltext/2019/02000/benchmarking_complications_associated_with.17.aspx)).
 - **MedCaseReasoning: Evaluating and learning diagnostic reasoning from clinical case reports**. Kevin Wu et.al. **arxiv**, **2025**, **Number of Citations: **None, ([pdf](./Papers/MedCaseReasoning_Evaluating_and_learning_diagnostic_reasoning_from_clinical_case_reports.pdf))([link](http://arxiv.org/abs/2505.11733v2)).
   - [https://huggingface.co/datasets/zou-lab/MedCaseReasoning](https://huggingface.co/datasets/zou-lab/MedCaseReasoning)

## <a id="Agent">Agent</a>
 - **Medagents: Large language models as collaborators for zero-shot medical reasoning**. X Tang et.al. **ACL (Findings)**, **2023**, **Number of Citations: **229, ([pdf](./Papers/Medagents_Large_language_models_as_collaborators_for_zero-shot_medical_reasoning.pdf))([link](https://arxiv.org/abs/2311.10537)).
