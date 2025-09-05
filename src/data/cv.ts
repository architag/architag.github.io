export const experiences = [
	{
		company: 'LocoNav',
		time: 'Jun 2021 - Aug 2024',
		title: 'Software Engineer',
		location: 'Gurugram, India',
		description: 'Developed high-throughput microservices, automated SaaS integrations, and IoT diagnostic tools while improving scalability, database performance, and system reliability.'
	},
	{
		company: 'Mibura Inc.',
		time: 'Jun 2025 - Aug 2025',
		title: 'Software Engineer Intern',
		location: 'Los Angeles, USA',
		description: 'Designed orchestration platform to automate server provisioning across heterogeneous hardware, with enhanced reliability through caching, RBAC, and rate limiting.'
	},
];

export const education = [
	{
		school: 'Guru Gobind Singh Indraprastha University',
		time: 'Aug 2017 - May 2021',
		degree: 'B.Tech in Computer Science and Engineering',
		location: 'New Delhi, India',
		description: 'Graduated with a CGPA of 8.96/10',
	},
	{
		school: 'New York University',
		time: 'Sep 2024 - May 2026',
		degree: 'M.S. in Computer Science',
		location: 'New York, USA',
		description: 'Currently pursuing AI Concentration with a GPA of 4.00/4.00',
	},
];

export const skills = [
	{
		title: 'Languages',
		description: 'Python, Java, C++, Ruby, JavaScript, SQL'
	},
	{
		title: 'Frameworks and Libraries',
		description: 'Spring Boot, Ruby on Rails, Flask, FastAPI,PyTorch, TensorFlow'
	},
	{
		title: 'Databases and Messaging',
		description: 'PostgreSQL, MySQL, MongoDB, Apache Kafka, Redis, ElasticSearch'
	},
	{
		title: 'Cloud and DevOps',
		description: 'AWS, Docker, Kubernetes, Jenkins, CI/CD'
	},
];

export const projects = [
	{
		title: "Factually Grounded Climate Reasoning in LLaMA",
		time: "May 2025",
		link: "https://github.com/architag/climate-reasoning-llama",
		abstract: "Designed and evaluated methods to improve causal reasoning of LLaMA in the climate domain. Compared fine-tuned, base, and RAG pipelines on structured causal QA and ClimateQA data, using metrics for lexical accuracy, semantic similarity, and entailment.",
		details: [
		  {
			heading: "Data Set",
			content: [
			  {
				itemHeading: "Climate Corpus:",
				itemContent: "IPCC reports and climate publications from the past five years, converted from PDFs to text for fine-tuning and retrieval."
			  },
			  {
				itemHeading: "Structured QA:",
				itemContent: "108 questions generated from 36 causal statements in climate texts, covering why, how, and relation-based prompts."
			  },
			  {
				itemHeading: "External Benchmark:",
				itemContent: "Ekimetrics ClimateQA dataset (3,430 public climate questions) used for out-of-distribution evaluation."
			  }
			]
		  },
		  {
			heading: "Approach",
			content: [
			  {
				itemHeading: "Fine-Tuning:",
				itemContent: "Adapted LLaMA 3.2-3B on climate-specific texts to strengthen domain grounding."
			  },
			  {
				itemHeading: "RAG Pipeline:",
				itemContent: "Constructed a MiniLM-based FAISS flat index over climate corpus and retrieved context for LLaMA answers."
			  },
			  {
				itemHeading: "Causal Evaluation:",
				itemContent: "Extracted causal frames from texts and generated targeted QA prompts to test reasoning fidelity."
			  },
			  {
				itemHeading: "Scoring:",
				itemContent: "Evaluated with lexical F1, semantic similarity, and entailment, aggregated into a combined score (0.3 F1, 0.4 semantic, 0.3 entailment)."
			  }
			]
		  },
		  {
			heading: "Result",
			content: [
			  {
				itemHeading: "Structured QA:",
				itemContent: "Fine-tuned LLaMA scored highest overall (0.628), compared to base (0.605) and RAG (0.580). Fine-tuned improved semantic similarity (0.742) and entailment (0.833), while RAG led in F1 (0.347)."
			  },
			  {
				itemHeading: "Consistency:",
				itemContent: "Fine-tuned model showed more reliable causal reasoning, while RAG’s performance varied with retrieval quality and prompt structure."
			  },
			  {
				itemHeading: "Ekimetrics Evaluation:",
				itemContent: "Fine-tuned model achieved higher cosine similarity on 46% of questions, vs. 26% for base; 28% tied."
			  }
			]
		  }
		]
	},	  
	{
		title: "Climate Retrieval-Augmented Generation (RAG)",
		time: "Apr 2025",
		link: "https://github.com/architag/mlsystems-spring-25/tree/main/assignment_3",
		abstract: "Built a Retrieval-Augmented Generation pipeline to improve climate-related question answering compared to fine-tuned models. Implemented FAISS-based indexing with multiple embeddings and search methods, and benchmarked retrieval latency, answer quality, and efficiency against a LLaMA 3.2-3B baseline.",
		details: [
		  {
			heading: "Data Set",
			content: [
			  {
				itemHeading: "",
				itemContent: "830 climate change reports and publications (747 train, 83 test), processed into 117K text chunks using recursive splitting (1000 characters, 50 overlap)."
			  }
			]
		  },
		  {
			heading: "RAG Pipeline",
			content: [
			  {
				itemHeading: "Indexing:",
				itemContent: "Generated embeddings with BGE-large and MiniLM; stored in FAISS indices (Flat, HNSW, IVF-flat, IVF-PQ)."
			  },
			  {
				itemHeading: "Retrieval:",
				itemContent: "Encoded queries with the same embedding model and retrieved top-5 nearest chunks for context construction."
			  },
			  {
				itemHeading: "Generation:",
				itemContent: "Combined retrieved context with user query and passed to LLaMA 3.2-3B for response generation (512-token limit)."
			  }
			]
		  },
		  {
			heading: "Result",
			content: [
			  {
				itemHeading: "Performance:",
				itemContent: "RAG averaged 47.6 ms per token vs. 43.1 ms for fine-tuned baseline. Embedding generation took ~30 min (BGE) vs. ~2.5 min (MiniLM)."
			  },
			  {
				itemHeading: "Answer Quality:",
				itemContent: "RAG produced concise, context-grounded answers, while the fine-tuned model gave more verbose responses of similar quality."
			  },
			  {
				itemHeading: "Index Trade-offs:",
				itemContent: "HNSW and PQ offered faster, memory-efficient retrieval; IVF index yielded higher-quality answers than Flat."
			  }
			]
		  }
		]
	},	  
	{
		title: 'Distributed Fine-Tuning LLaMA',
		time: 'Mar 2025',
		link: 'https://github.com/architag/mlsystems-spring-25/tree/main/assignment_2',
		abstract: `Implemented distributed training techniques to fine-tune a LLaMA 3.2-3B model across 2 GPUs. The focus is on improving training
		efficiency (time per epoch) through data parallelism, tensor parallelism, and pipeline parallelism, compared to single-GPU training.`,
		details: [
			{
				heading: 'Data Set',
				content: [
					{
						itemHeading: '',
						itemContent: 'The dataset consists of IPCC reports and climate change publications from the last five years, stored in PDF format. Processed these reports by extracting text, cleaning formatting issues, and chunking content into manageable segments.'
					}
				]
			},
			{
				heading: 'Distributed Training Techniques',
				content: [
					{
						itemHeading: 'Data Parallelism:',
						itemContent: `Split the training data across 2 GPUs while each GPU maintained a full copy of the model, 
						using PyTorch's native torchrun and Hugging Face's Trainer with DDP support.`
					},
					{
						itemHeading: 'Tensor Parallelism:',
						itemContent: `Splits weight matrices of large layers across multiple GPUs, where each GPU holds only part of the model's layers. Used the DeepSpeed library to implement tensor parallelism providing it with a ds_config.json
						file.`
					},
					{
						itemHeading: 'Pipeline Parallelism:',
						itemContent: `Assigns different layers of the model to different GPUs and processes micro-batches sequentially. Similar to Tensor Parallelism, used the DeepSpeed library to implement this.`
					}
				]
			},
			{
				heading: 'Result',
				content: [
					{
						itemHeading: 'Reduced Training Time:',
						itemContent: `Reduced training from ~1.5 hours per epoch to ~40 minutes per epoch.`
					},
					{
						itemHeading: 'Maintained model quality:',
						itemContent: `Achieved similar perplexity scores (~7.2) across all distributed methods.`
					},
					{
						itemHeading: 'Improved resource efficiency:',
						itemContent: `Reduced memory usage per GPU while utilizing multiple GPUs effectively.`
					}
				]
			}
		]
	},
	{
		title: 'Fine-Tuning LLaMA 3.2-3B',
		time: 'Feb 2025',
		link: 'https://github.com/architag/mlsystems-spring-25/tree/main/assignment_1',
		abstract: `Fine-tuned a LLaMA 3.2-3B model on climate change documents to create a specialized AI assistant for environmental topics.
		The project focused on memory-efficient training techniques to work within GPU constraints while maintaining model performance.
		The goal of the assignment is to increase the batch size of the training process as much as possible.`,
		details: [
			{
				heading: 'Data Set',
				content: [
					{
						itemHeading: '',
						itemContent: 'The dataset consists of IPCC reports and climate change publications from the last five years, stored in PDF format. Processed these reports by extracting text, cleaning formatting issues, and chunking content into manageable segments.'
					}
				]
			},
			{
				heading: 'Memory-Optimizations',
				content: [
					{
						itemHeading: 'LoRA (Low-Rank Adaptation):',
						itemContent: 'Reduces the number of trainable parameters by adding low-rank matrices to the pre-trained model weights. This allowed for efficient fine-tuning without updating the entire model.'
					},
					{
						itemHeading: '4-bit Quantization with QLoRA:',
						itemContent: 'Quantizes the model weights to 4 bits using BitsAndBytes library, significantly reducing the memory footprint.'
					},
					{
						itemHeading: 'Gradient Accumulation:',
						itemContent: 'Simulated a larger batch size without exceeding GPU memory limits. Maximum batch size achieved was 8. By accumulating gradients over 8 steps, the effective batch size was increased to 64.'
					},
					{
						itemHeading: 'Gradient Checkpointing:',
						itemContent: `Reduces memory usage by only storing a subset of the model's activations during backward pass, allowing for larger batch sizes without running out of memory`
					},
					{
						itemHeading: 'Mixed Precision Training:',
						itemContent: 'Used 16-bit floating point numbers instead of 32-bit to halve memory usage.'
					}
				]
			},
			{
				heading: 'Result',
				content: [
					{
						itemHeading: 'Maximum Batch Size:',
						itemContent: 'The maximum batch size achieved was 8, with gradient accumulation steps of 8, resulting in an effective batch size of 64. The memory consumption during training was ∼ 36GB. The batch size can be further squeezed to 10 with a memory usage of ~40GB.'
					},
					{
						itemHeading: 'Perplexity Score:',
						itemContent: 'The perplexity score on the test set was 7.25. The memory optimizations enabled training on a single GPU that would otherwise require multiple high-end GPUs or cloud resources.'
					},
				]
			}
		]
	},
	{
		title: 'State Prediction using JEPA',
		time: 'Dec 2024',
		link: 'https://github.com/architag/DL_Final_Project',
		abstract: `This project implements a Joint Embedding Predictive Architecture (JEPA) to learn a self-supervised
		world model in a simplified two-room environment to predict future agent states. The agent states are represented
		as embeddings based on the current state and action, without relying on raw observation reconstruction. The 
		architecture is inspired by energy-based methods and recent advances in representation learning.`,
		details: [
			{
				heading: 'Data Set',
				content: [
					{
						itemHeading: 'states.npy:',
						itemContent: '2-channel images (agent position + room layout) with shape [B, T, 2, H, W].'
					},
					{
						itemHeading: 'actions.npy:',
						itemContent: '2D movement vectors (dx, dy) with shape [B, T-1, 2].'
					},
					{
						itemHeading: 'locations.npy:',
						itemContent: 'Ground truth agent positions for evaluation.'
					}
				]
			},
			{
				heading: 'Architecture',
				content: [
					{
						itemHeading: 'Encoder:',
						itemContent: 'A Residual CNN with progressive layers and adaptive pooling, mapping 2-channel inputs to 256-dimensional embeddings.'
					},
					{
						itemHeading: 'Predictor:',
						itemContent: 'Projects actions to embedding space, then uses 4-head multi-head attention and an MLP with LayerNorm to predict future embeddings from current state and action.'
					},
					{
						itemHeading: 'Target Encoder:',
						itemContent: 'A momentum-updated clone of the encoder (BYOL-style) provides stable targets during training, preventing representation collapse.'
					}
				]
			},
			{
				heading: 'Training Losses',
				content: [
					{
						itemHeading: 'Invariance Loss:',
						itemContent: 'Ensures similar representations for augmented views.'
					},
					{
						itemHeading: 'Variance Loss:',
						itemContent: 'Prevents collapse by maintaining variance across dimensions.'
					},
					{
						itemHeading: 'Covariance Loss:',
						itemContent: 'Decorrelates features to avoid redundancy.'
					}
				]
			}
		]
	}
];
