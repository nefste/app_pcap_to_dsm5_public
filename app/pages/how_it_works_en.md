## What is the idea?

* **“Fuzzy”**: instead of hard yes/no thresholds you use $\color{#1f77b4}{\text{membership functions}}$ $\mu\in[0,1]$. A metric can be "a little unusual" (e.g. $\mu=0.4$) or "very unusual" ($\mu\approx1$).
* **“Additive”**: multiple fuzzy signals per criterion are weighted and summed (weights $\color{#2ca02c}{w}$ add up to 1). This yields a likelihood $\color{#ff7f0e}{L_k\in[0,1]}$ for criterion $\color{#9467bd}{k}$.
* **DSM‑Gate**: clinically consistent logic: a criterion counts as present when $\color{#ff7f0e}{L_k}$ is high on "most days" (e.g. ≥10/14) in a 14‑day window. An episode is **likely** when ≥5 criteria are present and at least one core symptom (Mood or Anhedonia) is included.

---

## The formula

For each criterion $\color{#9467bd}{k}$:

$$
\color{#ff7f0e}{L_k} = \color{#d62728}{\mathrm{clip}_{[0,1]}}\!\left(\sum_i \color{#2ca02c}{w_{k,i}} \, \color{#1f77b4}{\mu_{k,i}}(\color{#9467bd}{m_{k,i}})\right)
$$

* $\color{#9467bd}{m_{k,i}}$: your preprocessed metrics for criterion $\color{#9467bd}{k}$
* $\color{#1f77b4}{\mu_{k,i}(\cdot)}$: membership function (e.g. linear, triangular, trapezoid) returning [0,1]
* $\color{#2ca02c}{w_{k,i}}$: weight of metric $\color{#9467bd}{i}$ (per criterion sums to 1)
* $\color{#d62728}{\mathrm{clip}_{[0,1]}}$: constrain the sum to [0,1] (if the sum slightly exceeds 1)

**Important:** personalise/normalise first (e.g. z‑scores vs 30‑day baseline or relative deviation). This captures **changes** relative to the individual normal.

---

## Membership functions (intuitive)

Example: **linear** increase when something "keeps getting later":

$$
\mu(x)=
\begin{cases}
0, & x \le a\\
\dfrac{x-a}{b-a}, & a < x < b\\
1, & x \ge b
\end{cases}
$$

Example: **triangular** (peak at $b$, zero outside):

$$
\mu_{\text{tri}}(x;a,b,c)=
\begin{cases}
0, & x \le a\\[2pt]
\dfrac{x-a}{b-a}, & a< x \le b\\[4pt]
\dfrac{c-x}{c-b}, & b< x < c\\[4pt]
0, & x \ge c
\end{cases}
$$

Choose $a,b,c$ so that **0** means "not remarkable" and **1** means "clearly remarkable" — relative to the person.

---

## Example: Criterion **“Sleep disturbance”** (insomnia/hypersomnia)

We assume three signals on a given day:

* $\color{#1f77b4}{\mu_{\text{short\_sleep}}}$: short sleep duration (shorter → closer to 1)
* $\color{#1f77b4}{\mu_{\text{irregularity}}}$: irregularity (std. dev. of bedtimes across 14 days)
* $\color{#1f77b4}{\mu_{\text{late\_onset}}}$: late sleep onset (minutes after 22:00)

**Weights** (sum = 1):

$$
\color{#2ca02c}{w_{\text{short}}=0.5,\quad w_{\text{irreg}}=0.3,\quad w_{\text{late}}=0.2}
$$

### Define memberships

* **Short sleep duration** $h$ (linear; shorter → more abnormal):

  $$
  \mu_{\text{short}}(h)=
  \begin{cases}
  0, & h \ge 7.0\,\text{h}\\
  \dfrac{7.0-h}{7.0-5.0}, & 5.0 < h < 7.0\\
  1, & h \le 5.0
  \end{cases}
  $$

* **Irregularity** $\sigma$ (std. dev. of onset times over 14 days):

  $$
  \mu_{\text{irreg}}(\sigma)=
  \begin{cases}
  0, & \sigma \le 0.5\,\text{h}\\
  \dfrac{\sigma-0.5}{2.0-0.5}, & 0.5 < \sigma < 2.0\\
  1, & \sigma \ge 2.0
  \end{cases}
  $$

* **Late sleep onset** $m$ (minutes after 22:00):

  $$
  \mu_{\text{late}}(m)=
  \begin{cases}
  0, & m \le 30\,(22{:}30)\\
  \dfrac{m-30}{120-30}, & 30 < m < 120\,(\text{until midnight})\\
  1, & m \ge 120\,(\text{after 00{:}00})
  \end{cases}
  $$

> **Parameters** are examples – adapt them to your population.

### Plug in daily values

Assume on **day X**:

* Sleep duration $h = 5.5\,\text{h}$ → $\mu_{\text{short}} = \frac{1.5}{2} = 0.75$
* Irregularity $\sigma = 1.8\,\text{h}$ → $\mu_{\text{irreg}} \approx 0.87$
* Sleep onset $m = 110\,\text{min}$ (23:50) → $\mu_{\text{late}} \approx 0.89$

### Additively combine (apply weights)

$$
\begin{aligned}
\color{#ff7f0e}{L_{\text{sleep}}(X)}
&= 0.5\cdot 0.75 + 0.3\cdot 0.87 + 0.2\cdot 0.89\\
&= 0.375 + 0.261 + 0.178\\
&= 0.814 \Rightarrow \boxed{\color{#ff7f0e}{L_{\text{sleep}} \approx 0.81}}
\end{aligned}
$$

**Interpretation:** this day looks clearly sleep‑disturbed (value near 1).

---

## From day to DSM logic (14‑day window)

1. Repeat the calculation for each of the last 14 days → get 14 values $\color{#ff7f0e}{L_{\text{sleep}}(t)}$.
2. **Criterion present?** Define a threshold, e.g. $\color{#d62728}{\theta=0.7}$. Count how many days satisfy $L_{\text{sleep}}(t) \ge \theta$.
3. **Rule:** “present” if ≥10 of 14 days (or “most days” per DSM wording). Example: 11/14 days → Sleep disturbance = present.

> **Data quality (DQI):** if a day has poor data (e.g. little traffic), exclude it (or mark “unclear”) so artifacts are not rewarded.

---

## Second example: **Anhedonia** (loss of interest)

Take four metrics (drops vs personal baseline):

* $\mu_{\text{chat\_drop}}$: chat-session count ↓
* $\mu_{\text{domains\_drop}}$: unique domains ↓
* $\mu_{\text{prod\_drop}}$: productivity site hits ↓
* $\mu_{\text{up\_rate\_drop}}$: upstream chat rate ↓

**Weights:** $0.35, 0.30, 0.20, 0.15$

**Example day:**

* Chat −50% → $\mu=1.0$
* Domains −30% → $\mu=0.75$
* Productivity −60% → $\mu=1.0$
* Upstream −37.5% → $\mu\approx0.75$

$$
\begin{aligned}
\color{#ff7f0e}{L_{\text{anhedonia}}}
&= 0.35\cdot1.0 + 0.30\cdot0.75 + 0.20\cdot1.0 + 0.15\cdot0.75\\
&= 0.35 + 0.225 + 0.20 + 0.1125\\
&= 0.8875 \Rightarrow \boxed{\approx 0.89}
\end{aligned}
$$

Again, check across 14 days: if on ≥10 days the value ≥ $\color{#d62728}{\theta}$ → criterion present.

---

## Overall decision: DSM‑Gate

* **Per criterion:** determine presence via 14‑day rule.
* **Overall:** episode is likely if

  1. ≥5 criteria are present, and
  2. at least one core symptom (Mood or Anhedonia) is present.

> In PCAP data, Anhedonia is measurable and can count as a core symptom. Mood may be low-confidence or not evaluated.

---

## Why this is robust and explainable

* **Robust** because fuzzy: small fluctuations around thresholds do not flip decisions instantly.
* **Explainable** because additive: you see per criterion which metrics with which weights contributed how much.
* **Clinically consistent** because of the 14‑day window and DSM counting rule.
