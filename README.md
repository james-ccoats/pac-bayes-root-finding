# PAC-Bayes Root Finding

Numerical inversion of the binary KL divergence, the computational core of the
**PAC-Bayes-kl bound** (Maurer, 2004). Given a posterior's empirical error and
a complexity budget, it computes the tightest certified upper bound on the true
error by solving a one-dimensional root-finding problem.

Two solvers are provided and compared:

| Method | Idea | Convergence |
| --- | --- | --- |
| `kl_inv_bisection` | Bracket the root on `[p, 1)` and halve the interval | Linear, guaranteed |
| `kl_inv_newton` | Newton's method started at the Pinsker bound | Quadratic, monotone |

## Contents

- [Quick start](#quick-start)
- [The math](#the-math)
  - [1. Setting](#1-setting)
  - [2. The PAC-Bayes-kl bound](#2-the-pac-bayes-kl-bound)
  - [3. Turning the bound into a root-finding problem](#3-turning-the-bound-into-a-root-finding-problem)
  - [4. Properties of the binary KL divergence](#4-properties-of-the-binary-kl-divergence)
  - [5. Pinsker relaxation](#5-pinsker-relaxation)
  - [6. Bisection](#6-bisection)
  - [7. Newton's method](#7-newtons-method)
  - [8. Numerical considerations](#8-numerical-considerations)
- [Project layout](#project-layout)
- [Development](#development)
- [References](#references)

## Quick start

```bash
uv sync
```

```python
from pac_bayes_root_finding import calculate_e, kl_inv_newton, pinsker_bound

n = 10_000          # training samples
kl_qp = 20.0        # KL(Q || P) between posterior and prior over weights
delta = 0.05        # failure probability
p = 0.05            # empirical error of the posterior

e = calculate_e(kl_qp, n, delta)
print(kl_inv_newton(p, e))    # 0.0681  -> true error <= 6.81% w.p. >= 95%
print(pinsker_bound(p, e))    # 0.0876  -> looser closed-form relaxation
```

Run the solver comparison:

```bash
uv run python experiments/compare_methods.py
```

```
       n         e   Pinsker   bisection      newton    |diff|  bisect us  newton us
------------------------------------------------------------------------------------
     100   0.25991   0.41050  0.34748737  0.34748737   2.3e-14       12.9        3.3
    1000   0.02714   0.16650  0.11726243  0.11726243   8.2e-13       13.1        3.3
   10000   0.00283   0.08761  0.06810865  0.06810865   4.3e-13       13.2        4.0
  100000   0.00029   0.06213  0.05546628  0.05546628   1.5e-13       13.2        3.9
```

(Timings are per call and machine dependent.) Both solvers agree to ~1e-12,
and the exact inverse is substantially tighter than the Pinsker relaxation.

## The math

### 1. Setting

Consider a learning problem with data distribution $\mathcal{D}$, a hypothesis
space $\mathcal{H}$ (for example, neural network weights), and a bounded loss.
We draw $n$ i.i.d. samples $S$ and fix a **prior** $P$ over $\mathcal{H}$ chosen
*independently of $S$*. After seeing $S$, a learner outputs a **posterior**
$Q$ over $\mathcal{H}$ (it may depend on $S$ arbitrarily). Predictions are made
by sampling $h \sim Q$. Define for the 0-1 loss:

- the **empirical error** $\hat{L}_S(Q) = \mathbb{E}_{h\sim Q}\big[\tfrac1n \sum_i \mathbb{1}[h(x_i)\neq y_i]\big]$ (called `p` in the code),
- the **true error** $L(Q) = \mathbb{E}_{h\sim Q}\,\Pr_{(x,y)\sim\mathcal{D}}[h(x)\neq y]$ (the unknown quantity we want to bound; `q` in the code).

### 2. The PAC-Bayes-kl bound

Let $\mathrm{kl}(p\,\|\,q)$ be the KL divergence between Bernoulli
distributions,

$$
\mathrm{kl}(p\,\|\,q) = p\ln\frac{p}{q} + (1-p)\ln\frac{1-p}{1-q}.
$$

**Theorem (Maurer, 2004).** For $n \ge 8$ and any $\delta\in(0,1)$, with
probability at least $1-\delta$ over the draw of $S$, simultaneously for
**all** posteriors $Q$:

$$
\mathrm{kl}\big(\hat{L}_S(Q)\,\big\|\,L(Q)\big) \;\le\; \frac{\mathrm{KL}(Q\,\|\,P) + \ln\frac{2\sqrt{n}}{\delta}}{n}.
$$

The right-hand side is the quantity `e` in the code (`calculate_e`):

$$
e = \frac{\mathrm{KL}(Q\,\|\,P) + \ln\frac{2\sqrt{n}}{\delta}}{n}.
$$

It is a **complexity budget**: it grows with how far the posterior moved from
the prior, shrinks like $1/n$ with more data, and pays only a logarithmic
price for higher confidence ($\delta \to 0$).

### 3. Turning the bound into a root-finding problem

The theorem constrains $L(Q)$ only implicitly, through $\mathrm{kl}$. To get an
explicit certificate we invert it. Writing $p = \hat{L}_S(Q)$ and $q = L(Q)$,
the bound says $q$ lies in the set $\{q : \mathrm{kl}(p\,\|\,q) \le e\}$. The
**upper kl-inverse** is its largest element:

$$
\mathrm{kl}^{-1}(p, e) \;=\; \sup\{\, q \in [p, 1] : \mathrm{kl}(p\,\|\,q) \le e \,\}.
$$

so that, with probability at least $1-\delta$,

$$
L(Q) \;\le\; \mathrm{kl}^{-1}\!\big(\hat{L}_S(Q),\, e\big).
$$

There is no closed form for general $p$, but the supremum is attained at the
unique root $q^\star \in [p, 1)$ of

$$
g(q) \;=\; \mathrm{kl}(p\,\|\,q) - e \;=\; 0.
$$

This is exactly what `kl_inv_bisection` and `kl_inv_newton` compute.

(Special case $p=0$: $\mathrm{kl}(0\,\|\,q) = -\ln(1-q)$, so $q^\star = 1 - e^{-e}$ in closed form. This is used as a sanity check in the tests.)

### 4. Properties of the binary KL divergence

Fix $p\in[0,1]$ and view $\mathrm{kl}(p\,\|\,q)$ as a function of $q$.

**Derivative** (`kl_divergence_dq`). Differentiating directly,

$$
\frac{\partial}{\partial q}\mathrm{kl}(p\,\|\,q) = -\frac{p}{q} + \frac{1-p}{1-q} = \frac{q-p}{q(1-q)}
.
$$

This is zero at $q=p$ (the minimum, where $\mathrm{kl}=0$) and positive for
$q>p$, so $\mathrm{kl}$ is **strictly increasing** on $(p,1)$.

**Convexity.** The second derivative is

$$
\frac{\partial^2}{\partial q^2}\mathrm{kl}(p\,\|\,q) = \frac{p}{q^2} + \frac{1-p}{(1-q)^2} > 0,
$$

so $\mathrm{kl}$ is **strictly convex** in $q$.

**Boundary behaviour.** $\mathrm{kl}(p\,\|\,q)\to\infty$ as $q\to 1^-$ (when
$p<1$). Together with $g(p) = -e < 0$ and continuity, the intermediate value
theorem plus strict monotonicity gives that $g$ has **exactly one root** in
$(p, 1)$ for every $e>0$. The conventions $0\ln 0 = 0$ handle $p\in\{0,1\}$
(see `kl_divergence`).

### 5. Pinsker relaxation

Pinsker's inequality for Bernoulli distributions states
$\mathrm{kl}(p\,\|\,q) \ge 2(p-q)^2$. Hence $\mathrm{kl}(p\,\|\,q)\le e$
implies $2(q-p)^2 \le e$, giving the closed-form relaxation (`pinsker_bound`)

$$
\mathrm{kl}^{-1}(p, e) \;\le\; p + \sqrt{e/2} .
$$

It is cheap and always valid, but loose, especially when $p$ is small, where
the true inverse is bounded by $p + \sqrt{2pe} + 2e$, far below $p+\sqrt{e/2}$ (the
"fast rate" regime). The table above shows the gap (e.g. 0.0681 vs 0.0876).

Importantly for the solvers, $g(p+\sqrt{e/2}) \ge 0$, so the Pinsker bound
always lies **at or to the right of the root**. It serves as a free upper
bracket and as the Newton starting point.

### 6. Bisection

Maintain a bracket $[a, b]$ with $g(a)\le 0 < g(b)$. Start with
$a = p$ and $b = 1-10^{-15}$, then repeatedly evaluate $g$ at the midpoint and
keep the half that still contains the sign change. The bracket halves each
step, so reaching tolerance `tol` takes

$$
\left\lceil \log_2\frac{1-p}{\text{tol}} \right\rceil \approx 40 \text{ iterations for tol} = 10^{-12}.
$$

It is slow-but-certain: monotonicity of $g$ guarantees convergence for any
input, with no derivative needed. The returned left endpoint $a$ always satisfies
$\mathrm{kl}(p\,\|\,a)\le e$ and lies within `tol` of the root, so it can sit
marginally *below* $q^\star$ (by at most `tol`).

### 7. Newton's method

Newton's iteration for $g$ is

$$
q_{k+1} = q_k - \frac{g(q_k)}{g'(q_k)} = q_k - \frac{\mathrm{kl}(p\,\|\,q_k) - e}{(q_k-p)/(q_k(1-q_k))}.
$$

For a general function Newton's method can overshoot or diverge, but $g$ is
convex and increasing on $(p,1)$, which gives strong guarantees. If
$q_0 \ge q^\star$:

1. **Monotone descent.** Convexity means the tangent line at $q_k$ lies below
   $g$, so the tangent's root $q_{k+1}$ satisfies $g(q_{k+1}) \ge 0$, i.e.
   $q_{k+1}\ge q^\star$. Since $g(q_k)\ge0$ and $g'>0$, also $q_{k+1}\le q_k$.
2. **Stays in the domain.** The iterates are bounded below by $q^\star > p$, so
   they never leave $(p,1)$ and never hit the singularity at $q=p$ or $q=1$.
3. **Quadratic convergence.** Near the root the error satisfies
   $|q_{k+1}-q^\star| \approx \tfrac{g''(q^\star)}{2g'(q^\star)}\,|q_k-q^\star|^2$.

Starting from the Pinsker bound (Section 5) satisfies the precondition
$q_0\ge q^\star$ automatically, so the method is globally safe for this
problem, with 5-6 iterations in the benchmark above instead of 40, which is the roughly
3-4x speedup in the benchmark.

### 8. Numerical considerations

- **Stopping criterion.** `kl_inv_newton` stops on the residual $|g(q)|$ rather
  than the step size. Near $q = 1$ the derivative $g'$ is enormous, so steps
  can be tiny even when far from the root; a step-size test would stop too early.
- **Upper-side error.** Because Newton approaches the root from above, the
  returned value may sit marginally above $q^\star$. For a *valid* upper
  bound that is the conservative direction.
- **Clipping.** Both solvers cap $q$ at $1-10^{-15}$ so that $\ln(1-q)$ stays
  finite in double precision.
- **Edge cases.** $p=0$ and $p=1$ use the $0\ln 0=0$ convention. A zero budget
  $e=0$ returns $p$ to within solver tolerance (the $\mathrm{kl}$ is quadratic
  near $p$, so the accuracy in $q$ is about $\sqrt{\text{tol}}$).

## Project layout

```
src/pac_bayes_root_finding/
├── bounds.py          # complexity budget e and Pinsker relaxation
├── kl_divergence.py   # binary KL divergence and its q-derivative
└── root_finding.py    # bisection and Newton solvers for the kl-inverse
experiments/
└── compare_methods.py # accuracy / speed comparison of the solvers
tests/                 # pytest suite
```

## Development

```bash
uv sync                                             # install dependencies
uv run --group dev pytest                           # run the tests
uv run --group dev pytest --cov=pac_bayes_root_finding   # with coverage
```

## References

- A. Maurer. *A Note on the PAC Bayesian Theorem.* arXiv:cs/0411099, 2004.
- D. McAllester. *PAC-Bayesian Model Averaging.* COLT, 1999.
- M. Seeger. *PAC-Bayesian Generalisation Error Bounds for Gaussian Process Classification.* JMLR, 2002.
- G. K. Dziugaite and D. M. Roy. *Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks with Many More Parameters than Training Data.* UAI, 2017.
- M. Pérez-Ortiz, O. Rivasplata, J. Shawe-Taylor, C. Szepesvári. *Tighter Risk Certificates for Neural Networks.* JMLR, 2021.
