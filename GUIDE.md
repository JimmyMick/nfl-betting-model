# 🏈 Urban Zone Pick'em — User Guide

Welcome! This is a weekly NFL pick'em with a twist: you're not just competing
against your buddies — you're also up against a **stats model** (a computer
forecaster) and an **AI expert**. Pick winners each week, and we track who reads
the games best all season.

Here's everything you need to know.

---

## 🔑 Signing in

Click **"Log in with Descope"** and sign in with your email. Only invited emails
can get in, so use the address you gave Jim. Can't get in? Ping Jim to get added.

---

## 🗂️ The tabs at a glance

| Tab | What it's for |
|---|---|
| **Pick'em leaderboard** | The standings — who's winning, everyone vs the model |
| **Make picks** | Where you submit your winners + confidence each week |
| **Season tracker** | How the model itself is doing vs Vegas, week by week |
| **Weekly preview** | The model's read on this week's games (+ the AI's picks) |
| **📖 Guide** | This page |

---

## ✅ Making your picks (the main event)

1. Go to the **Make picks** tab.
2. For each game, click the team you think will **win**.
3. Set your **confidence** from **50 to 100**:
   - **50** = total coin flip, you have no real lean.
   - **100** = absolute lock, you'd bet your house.
   - Somewhere in between for everything else.
4. Click **Submit my picks**.

That's it. You can come back and re-submit any time **before the games kick off** —
your latest submission is the one that counts.

> **Why confidence matters:** you're scored two ways — how many games you get
> *right*, and how *honest* your confidence numbers are (see "calibration"
> below). So put real thought into the number, don't just slam 100 on everything.

---

## 🏆 Reading the leaderboard

Each player gets a row:

- **Record** — your wins-losses and win %.
- **vs Model** — *the headline number.* Your accuracy **minus the model's**, over
  the exact games you picked. **Positive (green) means you're beating the
  computer.** This is the brag metric.
- **Brier / Log loss** — your **calibration score** (lower is better). Plain
  English: if you say you're 90% sure, are you actually right about 90% of the
  time? Nailing your confidence levels scores well here; being wildly
  overconfident (100 on everything) or sandbagging (50 on everything) hurts you.

You're ranked by straight-up accuracy, but the calibration scores are where the
real sharps separate themselves.

---

## 📊 The Weekly preview — and one big caveat

This tab shows the **model's** take on each game:

- **Model** — the model's win probability for its favored team, e.g. `BUF 74%`.
- **Market** — what the Vegas betting line implies, e.g. `BUF 55%`.
- **Edge** — how much *more* the model likes its side than Vegas does. A
  **disagreement** measure.
- **Key driver** — the one factor pushing the model hardest (QB rating, roster
  talent, recent form, injuries, etc.).

> ⚠️ **A big "edge" is NOT a betting tip.** It just means the model disagrees with
> Vegas — and usually that means the *model* is wrong, not Vegas. This whole
> thing is a calibrated **forecaster**, not a tip sheet. (We tested it: betting
> these "edges" loses money. The market is really good.) Use it as a smart second
> opinion, not gospel.

**Two numbers worth knowing:**
- The model lands within ~1-3% of Vegas's accuracy — it's *as good as the market*,
  not better. That's the honest goal.
- **Top pick of the week** (in the Season tracker): the model's single
  most-confident game each week. Over six seasons those hit ~83%.

---

## 🤖 The AI expert

There's also an **AI expert** in the pool — a large language model that makes its
own picks. It plays **blind**: it never sees the model's numbers or the Vegas
line. It reasons from raw facts (records, recent form, injuries, weather) plus any
inside intel Jim feeds it, and it's encouraged to back underdogs when the spot is
right — not just rubber-stamp the favorites.

- **Before kickoff:** the AI's picks **and its reasoning** show up in the Weekly
  preview tab. Your picks stay hidden until games are graded — only the AI shows
  its hand early.
- **After grading:** look for the **"🤖 AI expert — why it picked what it did"**
  expander on the leaderboard to see its logic next to the ✓/✗ results.

Fair warning: it's a solid, opinionated expert — but it's reasoning over the same
info Vegas already knows, so don't expect it to run away with the title. Beating
the AI is very much on the table.

---

## 💡 Tips

- **Submit early, tweak late.** Get picks in, then adjust before kickoff as injury
  news drops.
- **Use confidence honestly.** A perfectly-calibrated 70%-confidence player can
  out-score a lucky 100%-on-everything player over a season.
- **Check the preview, trust your gut.** The model is a great second opinion, but
  the leaderboard rewards *your* read — that's the whole point.

Good luck, and may your upsets land. 🍀
