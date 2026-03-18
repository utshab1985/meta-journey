# CausalLift

### The ad measurement system that tells the truth.

---

## The Problem

Meta, Google, and every major ad platform has a dirty secret.

When they tell Nike "your ad caused 10,000 sales" — that number is wrong.

Not because they're lying. Because their measurement is broken.

Here's why:

The people who see ads on Instagram are already different.
They're more online. More tech savvy. More likely to buy anyway.

So when you compare:
- People who saw the ad → high purchase rate
- People who didn't → low purchase rate

You're not measuring the ad effect.
You're measuring the difference between types of people.

This is called **confounding**. And it costs advertisers billions every year.

---

## What CausalLift Does

CausalLift separates what the ad actually caused from what would 
have happened anyway.

Using causal inference and logistic regression, it finds the true 
incremental effect of any intervention — ad, feature, or product change.

---

## The Result

**Before CausalLift:**
> "Users who saw the ad were 7.66% more likely to buy."
> *(This number is wrong)*

**After CausalLift:**
> "Your ad made people 1.68x more likely to buy.
> But naturally tech-savvy users were already 3.36x more likely to buy 
> regardless of your ad.
> You are currently paying for both. You should only pay for the 1.68x."

---

## Who This Is For

- Data scientists tired of lying dashboards
- Advertisers who want to know what they actually caused
- Engineers building measurement systems at scale

---

## Quick Start
```python
# coming soon
```

---

## The Math

Built on logistic regression with causal controls.
Simulation tested on 1 million synthetic users.
More technical details in /docs (coming soon)

---

## Author

Utshab Chakravorty
Building in public. Day 2 of the journey.
