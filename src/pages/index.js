import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';

import styles from './index.module.css';

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
              Explore Curriculum
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
        <FeaturesSection />
        <CTASection />
      </main>
    </Layout>
  );
}
