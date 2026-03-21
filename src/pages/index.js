import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';

import styles from './index.module.css';

// ─── Appreciation Section ────────────────────────────────────────────────────
// Design Psychology applied:
//   • Reciprocity   — "This is free; your clap fuels more"
//   • Social Proof  — live-looking counter anchored at 312
//   • Variable Reward — randomised emoji particles on every clap
//   • Feedback Loop — progress bar + counter update instantly
//   • Peak-End Rule — positive emotional note at page end
function AppreciationSection() {
  const [userClaps, setUserClaps] = React.useState(0);
  const [isAnimating, setIsAnimating] = React.useState(false);
  const [particles, setParticles] = React.useState([]);
  const [mounted, setMounted] = React.useState(false);
  const BASE_CLAPS = 312;
  const MAX_CLAPS = 10;

  React.useEffect(() => {
    setMounted(true);
    const saved = parseInt(localStorage.getItem('aicohort_claps') || '0', 10);
    setUserClaps(Math.min(saved, MAX_CLAPS));
  }, []);

  const totalClaps = BASE_CLAPS + (mounted ? userClaps : 0);
  const progress = (userClaps / MAX_CLAPS) * 100;
  const isMaxed = userClaps >= MAX_CLAPS;

  const handleClap = () => {
    if (isMaxed) return;
    const newCount = userClaps + 1;
    setUserClaps(newCount);
    localStorage.setItem('aicohort_claps', newCount.toString());

    setIsAnimating(true);
    setTimeout(() => setIsAnimating(false), 350);

    const EMOJIS = ['⚡', '🧠', '🚀', '✨', '💡', '🎯', '🔥'];
    const emoji = EMOJIS[Math.floor(Math.random() * EMOJIS.length)];
    const x = Math.round((Math.random() - 0.5) * 100);
    const id = Date.now() + Math.random();
    setParticles(prev => [...prev, { id, emoji, x }]);
    setTimeout(() => setParticles(prev => prev.filter(p => p.id !== id)), 1200);
  };

  return (
    <section className={styles.appreciation}>
      <div className="container">
        <div className={styles.appreciationInner}>
          <div className={styles.appreciationHeader}>
            <span className={styles.appreciationLabel}>FOUND THIS HELPFUL?</span>
            <h2 className={styles.appreciationTitle}>Show Your Appreciation</h2>
            <p className={styles.appreciationSubtitle}>
              This entire curriculum is free. If it sparked something in you, give a clap — it takes one second and means a lot.
            </p>
          </div>

          <div className={styles.clapArea}>
            {/* Social Proof — anchoring effect */}
            <div className={styles.clapSocialProof}>
              <span className={styles.clapCount}>{mounted ? totalClaps.toLocaleString() : BASE_CLAPS.toLocaleString()}</span>
              <span className={styles.clapCountLabel}>learners appreciated this</span>
            </div>

            {/* Clap Button — variable reward via particles */}
            <div className={styles.clapButtonWrapper}>
              {particles.map(p => (
                <span
                  key={p.id}
                  className={styles.clapParticle}
                  style={{ '--particle-x': `${p.x}px` }}
                >
                  {p.emoji}
                </span>
              ))}
              <button
                className={clsx(
                  styles.clapButton,
                  isAnimating && styles.clapButtonActive,
                  isMaxed && styles.clapButtonMaxed,
                )}
                onClick={handleClap}
                aria-label={isMaxed ? 'Maximum claps given' : 'Clap to appreciate'}
                disabled={isMaxed}
              >
                <span className={styles.clapButtonIcon}>⚡</span>
                {userClaps > 0 && (
                  <span className={styles.clapUserCount}>+{userClaps}</span>
                )}
              </button>
            </div>

            {/* Progress Bar — feedback loop */}
            <div className={styles.clapProgress}>
              <div className={styles.clapProgressBar}>
                <div className={styles.clapProgressFill} style={{ width: `${progress}%` }} />
              </div>
              {isMaxed ? (
                <p className={styles.clapMaxMessage}>You gave all 10 claps — thank you! ⚡</p>
              ) : (
                <p className={styles.clapInstruction}>
                  {userClaps === 0
                    ? 'Click to clap — you can give up to 10'
                    : `${MAX_CLAPS - userClaps} clap${MAX_CLAPS - userClaps !== 1 ? 's' : ''} remaining`}
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

// ─── Creator Section ─────────────────────────────────────────────────────────
// Design Psychology applied:
//   • Authority + Likability — real name, face-like avatar, personal tagline
//   • Transparency          — direct contact info builds trust
//   • Contrast              — card stands out as the page's closing identity mark
function CreatorSection() {
  return (
    <section className={styles.creator}>
      <div className="container">
        <div className={styles.creatorCard}>
          <div className={styles.creatorLeft}>
            <img
              src="/img/bhagya.jpg"
              alt="Bhagya Rana"
              className={styles.creatorAvatar}
            />
          </div>
          <div className={styles.creatorRight}>
            <span className={styles.creatorBuilt}>Built by</span>
            <h3 className={styles.creatorName}>Bhagya Rana</h3>
            <p className={styles.creatorTagline}>Built with curiosity ⚡</p>
            <p className={styles.creatorBio}>
              AI practitioner & educator passionate about making LLM engineering accessible.
              Designed this curriculum to bridge the gap between theory and production-ready systems.
            </p>
            <div className={styles.creatorLinks}>
              <a href="mailto:bhagyarana2001@gmail.com" className={styles.socialLink} title="Send an email">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="16" height="16" aria-hidden="true">
                  <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/>
                  <polyline points="22,6 12,13 2,6"/>
                </svg>
                Email
              </a>
              <a href="https://bhagyarana.in/" className={styles.socialLink} title="Personal website" target="_blank" rel="noopener noreferrer">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="16" height="16" aria-hidden="true">
                  <circle cx="12" cy="12" r="10"/>
                  <line x1="2" y1="12" x2="22" y2="12"/>
                  <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
                </svg>
                Website
              </a>
              <a href="https://x.com/bhagya_rana" className={styles.socialLink} title="Twitter / X" target="_blank" rel="noopener noreferrer">
                <svg viewBox="0 0 24 24" fill="currentColor" width="16" height="16" aria-hidden="true">
                  <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-4.714-6.231-5.401 6.231H2.74l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
                </svg>
                Twitter / X
              </a>
              <a href="https://www.linkedin.com/in/bhagyarana/" className={styles.socialLink} title="LinkedIn" target="_blank" rel="noopener noreferrer">
                <svg viewBox="0 0 24 24" fill="currentColor" width="16" height="16" aria-hidden="true">
                  <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"/>
                  <rect x="2" y="9" width="4" height="12"/>
                  <circle cx="4" cy="4" r="2"/>
                </svg>
                LinkedIn
              </a>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

// Hero Section
function HeroSection() {
  return (
    <header className={styles.hero}>
      <div className={styles.heroBackground}>
        <div className={styles.heroGradient} />
        <div className={styles.heroDotGrid} />
        <div className={styles.heroGlow} />
      </div>
      <div className="container">
        <div className={styles.heroContent}>
          <div className={styles.heroBadge}>
            <span className={styles.heroBadgeIcon}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
              </svg>
            </span>
            AI Engineering Cohort
          </div>
          <h1 className={styles.heroTitle}>
            Build Production-Ready
            <br />
            <span className={styles.heroHighlight}>AI Agents</span>
          </h1>
          <p className={styles.heroSubtitle}>
            Go from prompts to scalable, stateful agent systems using LangChain, RAG,
            LangGraph, MCP, and real-world deployment patterns.
          </p>
          <div className={styles.heroMeta}>
            <div className={styles.heroMetaItem}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
                <line x1="16" y1="2" x2="16" y2="6" />
                <line x1="8" y1="2" x2="8" y2="6" />
                <line x1="3" y1="10" x2="21" y2="10" />
              </svg>
              5 Modules
            </div>
            <span className={styles.heroMetaDot} />
            <div className={styles.heroMetaItem}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <circle cx="12" cy="12" r="10" />
                <polyline points="12 6 12 12 16 14" />
              </svg>
              90 min each
            </div>
            <span className={styles.heroMetaDot} />
            <div className={styles.heroMetaItem}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                <polyline points="22 4 12 14.01 9 11.01" />
              </svg>
              Hands-on Labs
            </div>
          </div>
          <div className={styles.heroButtons}>
            <Link
              className={styles.heroCTA}
              to="/learn/onboarding">
              Start Learning
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="5" y1="12" x2="19" y2="12" />
                <polyline points="12 5 19 12 12 19" />
              </svg>
            </Link>
            <Link
              className={styles.heroSecondary}
              to="/learn/modules">
              Browse Modules
            </Link>
          </div>
          <p className={styles.heroCaption}>
            No prior AI experience required. Python basics recommended.
          </p>
        </div>
      </div>
      <div className={styles.scrollIndicator}>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </div>
    </header>
  );
}

// Cohort Outline Data
const cohortOutline = [
  {
    week: 'Week 1',
    number: '01',
    title: 'LangChain Fundamentals',
    description: 'Master the building blocks of LangChain including models, prompts, chains, and the LangChain Expression Language (LCEL).',
    bullets: [
      'OpenAI & model integration setup',
      'Prompt templates and dynamic prompting',
      'LCEL composition patterns',
      'Output parsers and structured data',
    ],
    tags: [
      { label: 'LangChain', color: 'purple' },
      { label: 'OpenAI', color: 'cyan' },
      { label: 'LCEL', color: 'rose' },
    ],
    link: '/learn/modules/module-1',
  },
  {
    week: 'Week 2',
    number: '02',
    title: 'Advanced LangChain & RAG',
    description: 'Deep dive into embeddings, vector stores, Retrieval-Augmented Generation, and complex chain architectures.',
    bullets: [
      'Text embeddings and similarity search',
      'ChromaDB vector store integration',
      'RAG pipeline from scratch',
      'Sequential and complex chains',
    ],
    tags: [
      { label: 'RAG', color: 'cyan' },
      { label: 'Embeddings', color: 'purple' },
      { label: 'Vector Stores', color: 'rose' },
    ],
    link: '/learn/modules/module-2',
  },
  {
    week: 'Week 3',
    number: '03',
    title: 'LangGraph Essentials',
    description: 'Build stateful, multi-step agent workflows with graph-based architecture, conditional routing, and memory.',
    bullets: [
      'Graph architecture fundamentals',
      'State management patterns',
      'Conditional flow routing',
      'Persistent memory systems',
    ],
    tags: [
      { label: 'LangGraph', color: 'purple' },
      { label: 'State Machines', color: 'cyan' },
      { label: 'Memory', color: 'rose' },
    ],
    link: '/learn/modules/module-3',
  },
  {
    week: 'Week 4',
    number: '04',
    title: 'Agents, Tools & MCP',
    description: 'Create intelligent agents with custom tools, Model Context Protocol integration, and multi-agent collaboration.',
    bullets: [
      'ReAct agent pattern implementation',
      'Custom tool development',
      'MCP server integration',
      'Multi-agent system design',
    ],
    tags: [
      { label: 'Agents', color: 'purple' },
      { label: 'MCP', color: 'cyan' },
      { label: 'Multi-Agent', color: 'rose' },
    ],
    link: '/learn/modules/module-4',
  },
  {
    week: 'Week 5',
    number: '05',
    title: 'Production Patterns',
    description: 'Scale and deploy AI agents with monitoring, testing, security best practices, and production architecture.',
    bullets: [
      'FastAPI deployment patterns',
      'Observability and monitoring',
      'Testing agent systems',
      'Security and guardrails',
    ],
    tags: [
      { label: 'Deployment', color: 'cyan' },
      { label: 'Testing', color: 'purple' },
      { label: 'Security', color: 'rose' },
    ],
    link: '/learn/modules/module-5',
    badge: 'Coming Soon',
  },
];

function CohortOutlineSection() {
  return (
    <section id="curriculum" className={styles.cohortOutline}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <span className={styles.sectionLabel}>CURRICULUM</span>
          <h2 className={styles.sectionTitle}>Cohort Outline</h2>
          <p className={styles.sectionSubtitle}>
            A structured learning path from AI fundamentals to production-ready agent systems
          </p>
        </div>
        <div className={styles.timeline}>
          {cohortOutline.map((item, index) => (
            <div key={index} className={styles.timelineItem}>
              <div className={styles.timelineLeft}>
                <span className={styles.timelineNumber}>{item.number}</span>
                <div className={styles.timelineDot} />
                {index < cohortOutline.length - 1 && (
                  <div className={styles.timelineLine} />
                )}
              </div>
              <div className={styles.timelineCard}>
                <div className={styles.timelineCardHeader}>
                  <span className={styles.weekLabel}>{item.week}:</span>
                  {item.badge && (
                    <span className={styles.timelineBadge}>{item.badge}</span>
                  )}
                </div>
                <h3 className={styles.timelineCardTitle}>{item.title}</h3>
                <p className={styles.timelineCardDesc}>{item.description}</p>
                <ul className={styles.timelineBullets}>
                  {item.bullets.map((bullet, i) => (
                    <li key={i} className={styles.timelineBullet}>{bullet}</li>
                  ))}
                </ul>
                <div className={styles.timelineTags}>
                  {item.tags.map((tag, i) => (
                    <span
                      key={i}
                      className={clsx(styles.tag, styles[`tag--${tag.color}`])}
                    >
                      {tag.label}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// Features Section
const features = [
  {
    title: 'Hands-on Learning',
    description: 'Every concept comes with Jupyter notebooks and code samples you can run immediately.',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
      </svg>
    ),
  },
  {
    title: 'Real-World Projects',
    description: 'Build agents that solve actual problems: RAG systems, tool-using agents, and more.',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    ),
  },
  {
    title: 'Multiple LLM Providers',
    description: 'Learn to work with OpenAI, Google AI, Anthropic, Groq, and more.',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
      </svg>
    ),
  },
  {
    title: 'Progressive Difficulty',
    description: 'Start with basics and progressively build to advanced multi-agent architectures.',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
      </svg>
    ),
  },
];

function FeaturesSection() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <span className={styles.sectionLabel}>WHY THIS TRAINING</span>
          <h2 className={styles.sectionTitle}>Built for Engineers</h2>
          <p className={styles.sectionSubtitle}>
            Designed for developers who want to build production AI agents
          </p>
        </div>
        <div className={styles.featuresGrid}>
          {features.map((feature, index) => (
            <div key={index} className={styles.featureCard}>
              <div className={styles.featureIcon}>{feature.icon}</div>
              <h3 className={styles.featureTitle}>{feature.title}</h3>
              <p className={styles.featureDescription}>{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// AI Engineering Section Data
const aiEngineeringModules = [
  { number: '00', title: 'Foundations', description: 'Python for ML, async APIs, data handling, and applied math essentials.' },
  { number: '01', title: 'LLM Fundamentals', description: 'How LLMs work: tokens, context windows, sampling, and the full inference pipeline.' },
  { number: '02', title: 'Transformer Internals', description: 'Attention, embeddings, KV cache, and why models behave the way they do.' },
  { number: '03', title: 'Prompting & Reasoning', description: 'Zero-shot, CoT, structured outputs, tool calling, and prompt injection defenses.' },
  { number: '04', title: 'RAG Systems', description: 'Retrieval-augmented generation: chunking, embedding, indexing, and reranking.' },
  { number: '05', title: 'Vector Databases', description: 'ANN search, HNSW, hybrid retrieval, quantization, and choosing the right DB.' },
  { number: '06', title: 'Model Optimization', description: 'Quantization, distillation, batching, and inference speed trade-offs.' },
  { number: '07', title: 'Fine-Tuning', description: 'LoRA, QLoRA, RLHF, data curation, and when fine-tuning beats prompting.' },
  { number: '08', title: 'Agents & System Design', description: 'ReAct, planning agents, tool use, multi-agent orchestration, and memory.' },
  { number: '09', title: 'Evaluation & Safety', description: 'Evals, benchmarks, red-teaming, hallucination mitigation, and guardrails.' },
  { number: '10', title: 'Multimodal Systems', description: 'Vision-language models, audio, structured data, and cross-modal pipelines.' },
  { number: '11', title: 'Production AI Systems', description: 'Deployment, monitoring, latency optimization, cost control, and CI/CD.' },
  { number: '12', title: 'Capstone Projects', description: 'Build four end-to-end systems: research assistant, support bot, code gen, and autonomous agent.' },
];

function AIEngineeringSection() {
  return (
    <section className={styles.aiEngineering}>
      <div className="container">
        <div className={styles.aiEngineeringDivider}>
          <span className={styles.aiEngineeringDividerLabel}>Also on this platform</span>
        </div>
        <div className={styles.sectionHeader}>
          <span className={clsx(styles.sectionLabel, styles.sectionLabelAmber)}>SELF-PACED</span>
          <h2 className={styles.sectionTitle}>AI Engineering Track</h2>
          <p className={styles.sectionSubtitle}>
            A 13-module concept-first curriculum covering everything from LLM internals to production deployment — learn on your own schedule.
          </p>
        </div>
        <div className={styles.aiEngineeringGrid}>
          {aiEngineeringModules.map((mod) => (
            <div key={mod.number} className={styles.aiEngineeringCard}>
              <span className={styles.aiEngineeringCardNumber}>{mod.number}</span>
              <h3 className={styles.aiEngineeringCardTitle}>{mod.title}</h3>
              <p className={styles.aiEngineeringCardDesc}>{mod.description}</p>
            </div>
          ))}
        </div>
        <div className={styles.aiEngineeringCTA}>
          <Link className={styles.aiEngineeringCTAButton} to="/learn/ai-engineering">
            Explore AI Engineering
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="5" y1="12" x2="19" y2="12" />
              <polyline points="12 5 19 12 12 19" />
            </svg>
          </Link>
        </div>
      </div>
    </section>
  );
}

// CTA Section
function CTASection() {
  return (
    <section className={styles.cta}>
      <div className="container">
        <div className={styles.ctaContent}>
          <h2 className={styles.ctaTitle}>Ready to Build Intelligent Agents?</h2>
          <p className={styles.ctaSubtitle}>
            Start your journey from AI fundamentals to production-ready agent systems
          </p>
          <Link
            className={styles.ctaCTA}
            to="/learn/onboarding">
            Begin Your Training
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="5" y1="12" x2="19" y2="12" />
              <polyline points="12 5 19 12 12 19" />
            </svg>
          </Link>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title="Learn AI Engineering"
      description="Master AI through structured cohort-based learning. Build intelligent agents with LangChain, LangGraph, and modern AI frameworks.">
      <HeroSection />
      <main>
        <CohortOutlineSection />
        <AIEngineeringSection />
        <FeaturesSection />
        <CTASection />
        <AppreciationSection />
        <CreatorSection />
      </main>
    </Layout>
  );
}
