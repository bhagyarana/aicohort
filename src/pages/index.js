import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';

import styles from './index.module.css';

// Hero Section
function HeroSection() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={styles.hero}>
      <div className={styles.heroBackground}>
        <div className={styles.heroGradient} />
        <div className={styles.heroGrid} />
      </div>
      <div className="container">
        <div className={styles.heroContent}>
          <div className={styles.heroBadge}>
            <span className={styles.heroBadgeDot} />
            Now with Module 5: Advanced Agent Architectures
          </div>
          <h1 className={styles.heroTitle}>
            Master <span className={styles.heroHighlight}>Agentic AI</span>
            <br />Development
          </h1>
          <p className={styles.heroSubtitle}>
            A comprehensive training program to build intelligent AI agents using LangChain,
            LangGraph, and modern AI frameworks. From fundamentals to production-ready systems.
          </p>
          <div className={styles.heroButtons}>
            <Link
              className={clsx('button button--primary button--lg', styles.heroButton)}
              to="/learn/onboarding">
              Start Learning
              <svg className={styles.buttonIcon} viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
            </Link>
            <Link
              className={clsx('button button--secondary button--lg', styles.heroButtonSecondary)}
              to="/learn/modules">
              Browse Modules
            </Link>
          </div>
          <div className={styles.heroStats}>
            <div className={styles.heroStat}>
              <span className={styles.heroStatNumber}>5</span>
              <span className={styles.heroStatLabel}>Learning Modules</span>
            </div>
            <div className={styles.heroStatDivider} />
            <div className={styles.heroStat}>
              <span className={styles.heroStatNumber}>15+</span>
              <span className={styles.heroStatLabel}>Hands-on Labs</span>
            </div>
            <div className={styles.heroStatDivider} />
            <div className={styles.heroStat}>
              <span className={styles.heroStatNumber}>1</span>
              <span className={styles.heroStatLabel}>Capstone Project</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

// Learning Path Section
const learningPath = [
  {
    step: '01',
    title: 'Onboarding & Setup',
    description: 'Get your environment ready with Python, API keys, and essential tools.',
    icon: '🚀',
    link: '/learn/onboarding',
  },
  {
    step: '02',
    title: 'Module-Based Learning',
    description: 'Progress through 5 comprehensive modules covering AI agent fundamentals to advanced patterns.',
    icon: '📚',
    link: '/learn/modules',
  },
  {
    step: '03',
    title: 'Agent Implementation',
    description: 'Apply your knowledge with 3 hands-on agent implementation patterns.',
    icon: '🤖',
    link: '/learn/agent-patterns',
  },
  {
    step: '04',
    title: 'Capstone Project',
    description: 'Build a production-ready AI agent system demonstrating your skills.',
    icon: '🏆',
    link: '/learn/capstone',
  },
];

function LearningPathSection() {
  return (
    <section className={styles.learningPath}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <h2 className={styles.sectionTitle}>Your Learning Journey</h2>
          <p className={styles.sectionSubtitle}>
            A structured path from beginner to building production-ready AI agents
          </p>
        </div>
        <div className={styles.pathGrid}>
          {learningPath.map((item, index) => (
            <Link key={index} to={item.link} className={styles.pathCard}>
              <div className={styles.pathCardStep}>{item.step}</div>
              <div className={styles.pathCardIcon}>{item.icon}</div>
              <h3 className={styles.pathCardTitle}>{item.title}</h3>
              <p className={styles.pathCardDescription}>{item.description}</p>
              <div className={styles.pathCardArrow}>
                <svg viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

// Modules Overview Section
const modules = [
  {
    number: 1,
    title: 'LangChain Fundamentals',
    description: 'Master the basics of LangChain including prompts, models, chains, and LCEL.',
    duration: '90 min',
    topics: ['OpenAI Integration', 'Prompt Templates', 'LCEL', 'Output Parsers'],
  },
  {
    number: 2,
    title: 'Advanced LangChain',
    description: 'Deep dive into embeddings, vector stores, RAG, and complex chains.',
    duration: '90 min',
    topics: ['Embeddings', 'Vector Stores', 'RAG Pipelines', 'Sequential Chains'],
  },
  {
    number: 3,
    title: 'LangGraph Essentials',
    description: 'Build stateful, multi-step agent workflows with LangGraph.',
    duration: '90 min',
    topics: ['Graph Architecture', 'State Management', 'Conditional Flows', 'Memory'],
  },
  {
    number: 4,
    title: 'Agents, Tools & MCP',
    description: 'Create intelligent agents with tools and Model Context Protocol.',
    duration: '90 min',
    topics: ['ReAct Agents', 'Custom Tools', 'MCP Integration', 'Multi-Agent Systems'],
  },
  {
    number: 5,
    title: 'Production Patterns',
    description: 'Scale and deploy agents with monitoring, testing, and best practices.',
    duration: '90 min',
    topics: ['Deployment', 'Monitoring', 'Testing', 'Security'],
    badge: 'Coming Soon',
  },
];

function ModulesSection() {
  return (
    <section className={styles.modules}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <h2 className={styles.sectionTitle}>Training Modules</h2>
          <p className={styles.sectionSubtitle}>
            Each module includes theory overview and hands-on implementation
          </p>
        </div>
        <div className={styles.modulesGrid}>
          {modules.map((module, index) => (
            <div key={index} className={styles.moduleCard}>
              <div className={styles.moduleCardHeader}>
                <div className={styles.moduleNumber}>{module.number}</div>
                {module.badge && (
                  <span className={styles.moduleBadge}>{module.badge}</span>
                )}
              </div>
              <h3 className={styles.moduleTitle}>{module.title}</h3>
              <p className={styles.moduleDescription}>{module.description}</p>
              <div className={styles.moduleDuration}>
                <svg className={styles.durationIcon} viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
                </svg>
                {module.duration}
              </div>
              <div className={styles.moduleTopics}>
                {module.topics.map((topic, i) => (
                  <span key={i} className={styles.moduleTopic}>{topic}</span>
                ))}
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
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
      </svg>
    ),
  },
  {
    title: 'Real-World Projects',
    description: 'Build agents that solve actual problems: RAG systems, tool-using agents, and more.',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    ),
  },
  {
    title: 'Multiple LLM Providers',
    description: 'Learn to work with OpenAI, Google AI, Anthropic, Groq, and more.',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
      </svg>
    ),
  },
  {
    title: 'Progressive Difficulty',
    description: 'Start with basics and progressively build to advanced multi-agent architectures.',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
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
          <h2 className={styles.sectionTitle}>Why This Training?</h2>
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
            className={clsx('button button--primary button--lg', styles.ctaButton)}
            to="/learn/onboarding">
            Begin Your Training
            <svg className={styles.buttonIcon} viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd" />
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
      title="Master AI Agent Development"
      description="Comprehensive training program for building intelligent AI agents with LangChain, LangGraph, and modern AI frameworks.">
      <HeroSection />
      <main>
        <LearningPathSection />
        <ModulesSection />
        <FeaturesSection />
        <CTASection />
      </main>
    </Layout>
  );
}
