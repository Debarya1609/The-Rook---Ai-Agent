# The Rook — Autonomous Marketing Intelligence Agent

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/5839aa64-557e-47fb-8440-8a6354c760fd" />



### The Rook AI is an autonomous multi-agent system designed to act as a full-time, in-house digital marketing strategist. It analyzes marketing performance, detects problems early, generates actionable plans, drafts client-ready communication, and simulates enterprise workflows — all without human intervention.

### Built with a powerful orchestration layer, parallel LLM agents, smart token budgeting, and custom tools, The Rook AI helps marketing teams work faster, communicate better, and make consistent data-backed decisions.

### From diagnosing campaign issues like low budget, bad creatives, ROAS drops, or spend spikes, to drafting clean email updates and creating structured task recommendations, The Rook delivers end-to-end marketing intelligence through an agentic architecture that is reliable, observable, and production-ready.

---

## 1. Problem Statement

Digital marketing teams often struggle with:

- Complex analytics across multiple ad platforms  
- Rapid fluctuations in ad performance, spend, conversions, and ROAS  
- Pressure to quickly resolve client issues  
- Overloaded teams unable to analyze data daily  
- Lack of dedicated strategists in smaller agencies  

These issues lead to:

- Late detection of critical problems  
- Slow or inconsistent client communication  
- Inefficient decision-making  

**The Rook** solves this by acting as an **autonomous in-house strategist**, powered by multi-agent reasoning, context-engineered LLM orchestration, and robust tool integrations.

---

## 2. Solution Overview

The Rook is a **multi-agent orchestration platform** designed to:

### Analyze six real marketing scenarios:
1. Bad Creatives  
2. Campaign Spend Spike  
3. Content Calendar Gap  
4. Developer Overload  
5. Low Budget  
6. Sudden Drop in ROAS  

### Generate:
- Structured JSON action plans  
- Email drafts  
- Task lists (via TaskAPI simulation)  
- Logs and metrics for observability  

### Provide:
- An interactive CLI (`Rook ai.py`)  
- Scenario token budgeting  
- Parallel email drafting workers  
- Merge agent to combine drafts  
- Automatic Gemini API key rotation (20-key load balancer)

This system simulates a full enterprise strategist workflow.

---

## 3. Why Agents?

Traditional scripts cannot:

- Perform multi-step reasoning  
- Pause for user approval  
- Autonomously call tools  
- Merge multi-draft outputs  
- Maintain state across multi-step processes  

**The Rook uses agents because marketing workflows require intelligent, adaptive decision flows.**

### Agents Used:
- **Strategy Agent** → Produces structured action plans  
- **Task Agent** → Converts plans into operational tasks  
- **Email Worker Agents** → Generate drafts in parallel  
- **Merge Agent** → Merges drafts into a final email  
- **Orchestrator** → Controls sequencing, logs, and approvals  

This demonstrates actual agentic architecture—not just prompting.

---

## 4. Architecture Overview

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/87cb3bf6-edf3-437f-96f4-d28496ea1b02" />



---

## 5. Key Hackathon Features Demonstrated

### Multi-Agent System
- Strategy Agent  
- Task Agent  
- Email Worker Agents  
- Merge Agent  
- Orchestrator  

### Parallel Agents
- Email generation uses 3–4 concurrent LLM workers.

### Sequential Agents
Scenario → Strategy → Task Generation → Email Draft → Merge → Output

### Custom Tools
| Tool | Description |
|------|-------------|
| `task_api.py` | Simulated enterprise task management tool |
| `email_api.py` | Parallel drafting + merge logic |
| `llm_client.py` | Gemini API key router with retry + fallback |
| `tune_token_budgets.py` | Auto token probing system |

### Long-Running Operations
- Multi-worker email drafting  
- Scenario analysis with retries  
- Automatic token tuning  

### Context Engineering
- Strict JSON formatting  
- Token budgets per scenario  
- Merge agent compresses context for final output  

### Sessions & State Handling
- Orchestrator manages flow state  
- Approval gate for low-confidence actions  
- Per-run logs stored for replay  

### Observability
Includes logs in `/logs/`:
- LLM raw responses  
- Token metadata  
- Worker results  
- Decision traces  

### Gemini Usage
- All reasoning powered by **Gemini 2.5 Flash**
- 20-key automatic load balancer ensures reliability  

---

## 6. How to Run

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Set Gemini API Keys

```bash
set MULTI_GEMINI_KEYS="KEY1,KEY2,KEY3,..."
```

### Step 3: Run the AI

```bash
python "Rook ai.py"

```

---


## 7. Example Output (Action Plan)


```json
{
  "actions": [
    {
      "action_type": "create_task",
      "details": {
        "task": "Audit creatives for LeadGen Oct"
      },
      "reason": "High CPA detected",
      "confidence": 0.6
    }
  ],
  "summary": "Creative fatigue detected. Investigation required."
}

```

## 8. Limitations & Future Improvements
- LLM still depends on free-tier Gemini quota
- Add real inputs solving & not just with demo inputs
- Add real email delivery via Gmail API 
- Build a web dashboard for agents and log viewing
- Add long-term memory for strategy refinement
- Deploy via Agent Engine or Cloud Run
- Future version will be updraded with more advancements

---

## 9. Conclusion
### The Rook is a production-style agentic system demonstrating:

- Real LLM reasoning
- Parallel + sequential agent orchestration
- Custom tools + context engineering
- Token budgeting and observability
- Interactive command-line interface

### This system automates decision workflows for marketing teams and shows the potential of autonomous agents inside enterprise environments.

---

## This project is submited to "Agents Intensive - Capstone Project" by Kaggle X Google 
### Members:
- Debarya Banerjee
- Tanish Badgal
- Salifu Alhassan
