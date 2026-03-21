/**
 * Agentic AI Training - Sidebar Configuration
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  trainingSidebar: [
    // Onboarding Section
    {
      type: 'category',
      label: 'Onboarding & Setup',
      collapsed: true,
      link: {
        type: 'doc',
        id: 'onboarding/index',
      },
      items: [
        'onboarding/training-guidelines',
        'onboarding/prerequisites',
        'onboarding/python-setup',
        'onboarding/api-keys-setup',
      ],
    },

    // Modules Section
    {
      type: 'category',
      label: 'Learning Modules',
      collapsed: true,
      link: {
        type: 'doc',
        id: 'modules/index',
      },
      items: [
        // Module 1
        {
          type: 'category',
          label: 'Module 1: LangChain Fundamentals',
          link: {
            type: 'doc',
            id: 'modules/module-1/index',
          },
          items: [
            'modules/module-1/overview',
            'modules/module-1/hands-on',
            'modules/module-1/resources',
          ],
        },
        // Module 2
        {
          type: 'category',
          label: 'Module 2: Advanced LangChain',
          link: {
            type: 'doc',
            id: 'modules/module-2/index',
          },
          items: [
            'modules/module-2/overview',
            'modules/module-2/hands-on',
            'modules/module-2/resources',
          ],
        },
        // Module 3
        {
          type: 'category',
          label: 'Module 3: LangGraph',
          link: {
            type: 'doc',
            id: 'modules/module-3/index',
          },
          items: [
            'modules/module-3/overview',
            'modules/module-3/hands-on',
            'modules/module-3/resources',
          ],
        },
        // Module 4
        {
          type: 'category',
          label: 'Module 4: Agents, Tools & MCP',
          link: {
            type: 'doc',
            id: 'modules/module-4/index',
          },
          items: [
            'modules/module-4/overview',
            'modules/module-4/hands-on',
            'modules/module-4/resources',
          ],
        },
        // Module 5
        {
          type: 'category',
          label: 'Module 5: Production Patterns',
          link: {
            type: 'doc',
            id: 'modules/module-5/index',
          },
          items: [
            'modules/module-5/overview',
            'modules/module-5/hands-on',
            'modules/module-5/resources',
          ],
        },
      ],
    },

    // Agent Implementation Series
    {
      type: 'category',
      label: 'Agent Implementation Series',
      collapsed: true,
      link: {
        type: 'doc',
        id: 'agent-patterns/index',
      },
      items: [
        {
          type: 'category',
          label: 'Pattern 1: ReAct Agent',
          link: {
            type: 'doc',
            id: 'agent-patterns/pattern-1/index',
          },
          items: [
            'agent-patterns/pattern-1/hands-on',
            'agent-patterns/pattern-1/review',
          ],
        },
        {
          type: 'category',
          label: 'Pattern 2: Tool-Using Agent',
          link: {
            type: 'doc',
            id: 'agent-patterns/pattern-2/index',
          },
          items: [
            'agent-patterns/pattern-2/hands-on',
            'agent-patterns/pattern-2/review',
          ],
        },
        {
          type: 'category',
          label: 'Pattern 3: Multi-Agent System',
          link: {
            type: 'doc',
            id: 'agent-patterns/pattern-3/index',
          },
          items: [
            'agent-patterns/pattern-3/hands-on',
            'agent-patterns/pattern-3/review',
          ],
        },
      ],
    },

    // Capstone Project
    {
      type: 'category',
      label: 'Capstone Project',
      collapsed: true,
      link: {
        type: 'doc',
        id: 'capstone/index',
      },
      items: [
        'capstone/requirements',
        'capstone/implementation',
        'capstone/review',
      ],
    },
  ],

  aiEngineeringSidebar: [
    {
      type: 'doc',
      id: 'ai-engineering/index',
      label: 'AI Engineering Track',
    },
    {
      type: 'category',
      label: 'Module 0: Foundations',
      collapsed: true,
      link: { type: 'doc', id: 'ai-engineering/module-0-foundations/index' },
      items: [
        'ai-engineering/module-0-foundations/overview',
        'ai-engineering/module-0-foundations/hands-on',
        'ai-engineering/module-0-foundations/resources',
      ],
    },
    {
      type: 'category',
      label: 'Module 1: LLM Fundamentals',
      collapsed: true,
      link: { type: 'doc', id: 'ai-engineering/module-1-llm-fundamentals/index' },
      items: [
        'ai-engineering/module-1-llm-fundamentals/overview',
        'ai-engineering/module-1-llm-fundamentals/hands-on',
        'ai-engineering/module-1-llm-fundamentals/resources',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Transformer Internals',
      collapsed: true,
      link: { type: 'doc', id: 'ai-engineering/module-2-transformer-internals/index' },
      items: [
        'ai-engineering/module-2-transformer-internals/overview',
        'ai-engineering/module-2-transformer-internals/hands-on',
        'ai-engineering/module-2-transformer-internals/resources',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: Prompting & Reasoning',
      collapsed: true,
      link: { type: 'doc', id: 'ai-engineering/module-3-prompting-reasoning/index' },
      items: [
        'ai-engineering/module-3-prompting-reasoning/overview',
        'ai-engineering/module-3-prompting-reasoning/hands-on',
        'ai-engineering/module-3-prompting-reasoning/resources',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: RAG Systems',
      collapsed: true,
      link: { type: 'doc', id: 'ai-engineering/module-4-rag-systems/index' },
      items: [
        'ai-engineering/module-4-rag-systems/overview',
        'ai-engineering/module-4-rag-systems/hands-on',
        'ai-engineering/module-4-rag-systems/resources',
      ],
    },
    {
      type: 'category',
      label: 'Module 5: Vector Databases',
      collapsed: true,
      link: { type: 'doc', id: 'ai-engineering/module-5-vector-databases/index' },
      items: [
        'ai-engineering/module-5-vector-databases/overview',
        'ai-engineering/module-5-vector-databases/hands-on',
        'ai-engineering/module-5-vector-databases/resources',
      ],
    },
    {
      type: 'category',
      label: 'Module 6: Model Optimization',
      collapsed: true,
      link: { type: 'doc', id: 'ai-engineering/module-6-model-optimization/index' },
      items: [
        'ai-engineering/module-6-model-optimization/overview',
        'ai-engineering/module-6-model-optimization/hands-on',
        'ai-engineering/module-6-model-optimization/resources',
      ],
    },
    {
      type: 'category',
      label: 'Module 7: Fine-Tuning',
      collapsed: true,
      link: { type: 'doc', id: 'ai-engineering/module-7-fine-tuning/index' },
      items: [
        'ai-engineering/module-7-fine-tuning/overview',
        'ai-engineering/module-7-fine-tuning/hands-on',
        'ai-engineering/module-7-fine-tuning/resources',
      ],
    },
    {
      type: 'category',
      label: 'Module 8: Agents & System Design',
      collapsed: true,
      link: { type: 'doc', id: 'ai-engineering/module-8-agents-system-design/index' },
      items: [
        'ai-engineering/module-8-agents-system-design/overview',
        'ai-engineering/module-8-agents-system-design/hands-on',
        'ai-engineering/module-8-agents-system-design/resources',
      ],
    },
    {
      type: 'category',
      label: 'Module 9: Evaluation & Safety',
      collapsed: true,
      link: { type: 'doc', id: 'ai-engineering/module-9-evaluation-safety/index' },
      items: [
        'ai-engineering/module-9-evaluation-safety/overview',
        'ai-engineering/module-9-evaluation-safety/hands-on',
        'ai-engineering/module-9-evaluation-safety/resources',
      ],
    },
    {
      type: 'category',
      label: 'Module 10: Multimodal Systems',
      collapsed: true,
      link: { type: 'doc', id: 'ai-engineering/module-10-multimodal/index' },
      items: [
        'ai-engineering/module-10-multimodal/overview',
        'ai-engineering/module-10-multimodal/hands-on',
        'ai-engineering/module-10-multimodal/resources',
      ],
    },
    {
      type: 'category',
      label: 'Module 11: Production AI Systems',
      collapsed: true,
      link: { type: 'doc', id: 'ai-engineering/module-11-production-systems/index' },
      items: [
        'ai-engineering/module-11-production-systems/overview',
        'ai-engineering/module-11-production-systems/hands-on',
        'ai-engineering/module-11-production-systems/resources',
      ],
    },
    {
      type: 'category',
      label: 'Module 12: Capstone Projects',
      collapsed: true,
      link: { type: 'doc', id: 'ai-engineering/module-12-capstone/index' },
      items: [
        'ai-engineering/module-12-capstone/project-1-research-assistant',
        'ai-engineering/module-12-capstone/project-2-support-automation',
        'ai-engineering/module-12-capstone/project-3-code-generation',
        'ai-engineering/module-12-capstone/project-4-autonomous-agent',
      ],
    },
  ],
};

module.exports = sidebars;
