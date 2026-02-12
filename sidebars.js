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
};

module.exports = sidebars;
