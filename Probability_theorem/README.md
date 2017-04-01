
# Markov Inequality
If $X$ is a $\textbf{nonnegative}$ random variable
$$
E[X] = \int_{0}^{a} tf_X(t) dt + \int_{a}^{\infty} tf_X(t) dt
\geq \int_{a}^{\infty} tf_X(t) dt
$$

$$
\geq \int_{a}^{\infty} af_X(t)ds = a P[X\geq a]
$$
  
$$
P[X \geq a]\leq \frac{E[X]}{a}
$$

# Chebyshev inequality
$E[X]=m, VAR[X]=\sigma^2$  
$Y = (X-m)^2$  
by Markov inequailty
$$
P[Y\geq a^2] \leq \frac{E[Y]}{a^2} = \frac{E[X-m]^2}{a^2} = \frac{\sigma ^2}{a^2}
$$
  
$$
P[|X-m|\geq a] \leq \frac{\sigma ^2}{a^2}
$$


# Laws of Large numbers
$X_i$ iid random variables
$E[X_i] = \mu, E[(X_i-\mu)^2]=\sigma^2$  
$$M_n = \frac{1}{n}\sum_{j=1}^{n} X_j$$  
$$E[M_n] = \mu, E[(M_n-\mu)^2]=\frac{\sigma^2}{n}$$ 
  
$$
P[|M_n-\mu|\geq \epsilon]\leq \frac{\sigma^2}{n\epsilon^2}
$$   
$$
P[|M_n-\mu| < \epsilon]\geq 1-\frac{\sigma^2}{n\epsilon^2}
$$
  
$$
\lim_{n \to \infty}P[|M_n-\mu|<\epsilon]=1
$$

# Characteristic function
$\Phi_X(\omega) = E[e^{j\omega X}]$  
#### 1) gaussian normal distribution case
Probability distribution function
$$f_X(x) = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-m)^2}{2\sigma^2}}$$
Characteristic funtion
$$\Phi_X(\omega) = E[e^{j\omega X}]$$
  
$$=\int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-m)^2}{2\sigma^2}+j\omega x} dx $$
  
$$=e^{j\omega m-\omega^2\sigma^2/2}\int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-(m+j\omega \sigma^2))^2}{2\sigma^2}} dx $$  
  
$$=e^{j\omega m-\omega^2\sigma^2/2}$$

# Central limit theorem
$X_i$ iid random variables   
$E[X_i] = \mu, E[(X_i-\mu)^2]=\sigma^2$
$$
Z_n=\frac{\sum_{k=1}^{n} (X_k-\mu)}{\sqrt{n}}
$$
Then $Z_n$ follows the gaussian distribution mean 0 and variance $\sigma^2$   
or $Z_n \sim N(0,\sigma^2)$
  
$\textit{proof)}$  
Characteristic function of $Z_n$  
$$\Phi_{Z_n}(\omega) = E[e^{j\omega \frac{\sum_{k=1}^{n} (X_k-\mu)}{\sqrt{n}}}]=
E[\prod_{k=1}^{n}e^{j\omega \frac{(X_k-\mu)}{\sqrt{n}}}]$$(iid condition)  
$$=\prod_{k=1}^{n}E[e^{j\omega \frac{(X_k-\mu)}{\sqrt{n}}}] = (E[e^{j\omega \frac{(X_k-\mu)}{\sqrt{n}}}])^n $$  
$$=(1+j\omega \frac{(X_k-\mu)}{\sqrt{n}}-\omega^2 \frac{(X_k-\mu)^2}{2n}+E[R(\omega)])^n$$ 
$$\lim_{n\to \infty}\Phi_{Z_n}(\omega)=(1-\frac{\omega^2\sigma^2}{2n})^n= e^{\frac{-\omega^2\sigma^2}{2}}$$  
$$\lim_{n\to \infty} P(Z_n = z) = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{z^2}{2\sigma^2}}$$

# Gamma function
$\Gamma(z)=\int_{0}^{\infty}e^{-x}x^{z-1}dx$  
$\Gamma(z+1)=\int_{0}^{\infty}e^{-x}x^{z}dx = -e^{-x}x^{z}|_{0}^{\infty}+z\int_{0}^{\infty}e^{-x}x^{z-1}dx=z\Gamma(z)$ 
  
   
$$\Gamma(\frac{1}{2})=\int_{0}^{\infty}\frac{e^{-x}}{\sqrt{x}}dx = \int_{0}^{\infty} 2e^{-y^2} dy$$  
  
(change of variable $x=y^2$) 
  
    
$$\Gamma(\frac{1}{2})^2=\int_{0}^{\infty} 2e^{-x^2} dx \cdot \int_{0}^{\infty} 2e^{-y^2} dy= \int_{0}^{\infty} \int_{0}^{\infty} 4e^{-x^2-y^2} dxdy$$(change of variable
$x=rcos\theta, y=rsin\theta$)
  
    
$$=\int_{0}^{\frac{\pi}{2}} \int_{0}^{\infty} 4e^{-r^2} rdrd\theta =\int_{0}^{\frac{\pi}{2}}  -2e^{-r^2} |_{0}^{\infty}d\theta = \pi$$

  
$$\Gamma(\frac{1}{2}) = \sqrt{\pi}$$

$$\Gamma(1)=1, \Gamma(n)=(n-1)!$$($n$ is integer)

# Beta function

$B(x,y) = \int_0^{1} t^{x-1}(1-t)^{y-1}dt$
$$\Gamma(x)\Gamma(y) = \int_0^{\infty} u^{x-1}e^{-u}du\cdot\int_0^{\infty} v^{y-1}e^{-v}dv$$  
  
$$= \int_0^{\infty} \int_0^{\infty} u^{x-1}v^{y-1}e^{-(u+v)}dudv$$  
Change of variable $u=zt, v=z(1-t)$

$$= \int_0^{\infty} \int_0^{1} (zt)^{x-1}(z(1-t))^{y-1}e^{-z}zdtdz$$  
  
$$= \int_0^{\infty} \int_0^{1} z^{x+y-1}t^{x-1}(1-t)^{y-1}e^{-z}dtdz$$ 
  
 $$= \int_0^{\infty}z^{x+y-1}e^{-z} dz \cdot\int_0^{1} t^{x-1}(1-t)^{y-1}dt$$
 
$$= \Gamma(x+y)B(x,y)$$

$$ B(x,y)=\frac{\Gamma(x)\Gamma(y)}{\Gamma(x+y)}$$

# Chi square proof
$z_i \sim N(0,1)$, and 
$$P(z_i)=\frac{1}{\sqrt{2\pi}}e^{-\frac{z_i^2}{2}}$$
 
Random variable $X$  
$$ X = \sum_{i=0}^{n}z_i^2$$

$$ P(X \leq x)= \int_{z_1^2+\cdots+z_n^2 \leq x} \prod_{i=1}^{n}P(z_i)dz_1\cdots dz_n$$
  (for $x\geq0$)
  
$$ P(x-dx \leq X \leq x)= \frac{1}{(2\pi)^{n/2}}\int_{x-dx \leq z_1^2+\cdots+z_n^2 \leq x} e^{-\frac{z_1^2+\cdots+z_n^2}{2}}dz_1\cdots dz_n$$  

$$ = e^{-\frac{x}{2}}\frac{1}{(2\pi)^{n/2}}\int_{x-dx \leq z_1^2+\cdots+z_n^2 \leq x} dz_1\cdots dz_n$$  

 Let  
$$ f_{n}(x)= \int_{z_1^2+\cdots+z_n^2 \leq x} dz_1\cdots dz_n$$  
  

$\textbf{Property1}$  
change of variable $z_i = \sqrt{x}\cdot y_i$
$$ f_{n}(x)= x^{n/2}\int_{y_1^2+\cdots+y_n^2 \leq 1} dy_1\cdots dy_n = x^{n/2}f_n(1)$$ 

$\textbf{Property2}$
$$ f_{n}(1)= \int_{-1}^{1}\int_{z_1^2+\cdots+z_{n-1}^2 \leq 1-z_n^2} dz_1\cdots dz_{n-1} dz_n$$  
  
$$ f_{n}(1)= \int_{-1}^{1}f_{n-1}(1-z_n^2)dz_n$$

$$ f_{n}(1)= f_{n-1}(1)\int_{-1}^{1}(1-z_n^2)^{(n-1)/2}dz_n$$

$$ = f_{n-1}(1)\frac{\Gamma(\frac{n-1}{2}+1)\Gamma(\frac{1}{2})}{\Gamma(\frac{n}{2}+1)}$$
$$ = \frac{f_{1}(1)}{2}\frac{\Gamma(\frac{1}{2})^{n}}{\Gamma(\frac{n}{2}+1)}$$
  
$$ = \frac{\pi^{n/2}}{\Gamma(\frac{n}{2}+1)}$$

$\textbf{Lemma}$  
$$\int_{-1}^{1}(1-x^2)^{n/2}dx = 2\int_{0}^{1}(1-x^2)^{n/2}dx$$  
change of variable $x = \sqrt{y}$   
$$= \int_{0}^{1}(1-y)^{n/2}y^{-1/2}dy = \frac{\Gamma(\frac{n}{2}+1)\Gamma(\frac{1}{2})}{\Gamma(\frac{n+1}{2}+1)}$$  

$$ P(X=x)=\frac{P(x-dx \leq X \leq x)}{dx}=
e^{-\frac{x}{2}}\frac{1}{(2\pi)^{n/2}}
\frac{f_n(x)-f_n(x-dx)}{dx}$$  

$$= e^{-\frac{x}{2}}\frac{1}{(2\pi)^{n/2}}\frac{d}{dx}f_n(x)= e^{-\frac{x}{2}}\frac{1}{(2\pi)^{n/2}}\frac{d}{dx}(x^{n/2}f_n(1))$$
  
$$ = \frac{1}{2^{n/2}\Gamma(\frac{n}{2})}x^{n/2-1}e^{-\frac{x}{2}}$$
