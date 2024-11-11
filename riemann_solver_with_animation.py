import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button


class ExactRiemannSolver(): # according to Toro chapter 4
	
	def __init__(self, L, nx, CFL, rho_L, u_L, p_L, rho_R, u_R, p_R, gamma, t):
		
		self.set_init_state(rho_L, u_L, p_L, rho_R, u_R, p_R, gamma, t)
		L = L  # domain [-L/2,L/2], shock initially at x=0
		nx = nx   # resolution
		self.CFL = CFL
		self.x   = np.linspace(-L/2,L/2,nx)
		self.rho = np.zeros(nx)
		self.u  = np.zeros(nx)
		self.p   = np.zeros(nx)
		self.T = np.zeros(nx)
		
		
	def set_init_state(self, rho_L, u_L, p_L, rho_R, u_R, p_R, gamma, t):
		"""
		Set Left and Right Initial States
		"""
		self.rho_L = rho_L
		self.u_L  = u_L
		self.p_L   = p_L

		self.rho_R = rho_R
		self.u_R  = u_R
		self.p_R   = p_R
		
		self.gamma = gamma
		self.t     = t
		
		self.p_star  = 0      # pressure solution in star region
		self.u_star = 0      # velocity solution in star region
		self.success = False  # solve succesful?
		
		# sound speeds
		self.a_L = np.sqrt(gamma*p_L/rho_L)
		self.a_R = np.sqrt(gamma*p_R/rho_R)

		
	def f_and_df(self, p, rho_k, p_k, gamma):
		"""
        Evaluate functions to solve for pressure in Newton-Raphson iterator
        Toro 4.2, 4.3.1
        k is L or R
        """
		if p <= p_k:
			c_k = np.sqrt(gamma * p_k / rho_k) # sound speed
			q = p/p_k
			f  = (2*c_k / (gamma-1)) * (q**((gamma-1)/(2*gamma)) - 1.0)
			df = (1/rho_k*p_k) * q**((-1-gamma)/(2*gamma))
		else:
			# Shock Wave
			Ak = 2/((gamma+1)*rho_k)
			Bk = (gamma-1)*p_k/(gamma+1)
			f  = (p - p_k) * np.sqrt(Ak/(p+Bk))
			df = np.sqrt(Ak/(p+Bk)) * (1 - 0.5*(p - p_k)/(p+Bk))
			
		return (f, df)		
		
	def calc_star_p_u(self):
		"""
		Compute solution for pressure & velocity in the star region
		Iterative Scheme for Finding the Pressure, Toro 4.3.2
        Solution for equation f = f_L + f_R + delta_u = 0
        is P(n) = P(n-1) - f(Pn-1)/df(Pn-1)
		"""

		tolerance = 1.0e-8
		max_iter  = 100

		p_prev = (self.p_L+self.p_R)/2
		# p_prev = self.guess_P()
		delta_u= self.u_R - self.u_L

		# compute pressure in star region via Newton-Raphson iteration
		for i in np.arange(max_iter):
			(f_L, df_L) = self.f_and_df(p_prev, self.rho_L, self.p_L, self.gamma)
			(f_R, df_R) = self.f_and_df(p_prev, self.rho_R, self.p_R, self.gamma)
			
			p = p_prev - (f_L + f_R + delta_u) / (df_L + df_R)
			change = 2.0 * abs((p - p_prev)/(p + p_prev)) # stop condition
			
			if change < tolerance:
				break	
				
			if (p < 0.0): 
				p = tolerance
				
			p_prev = p
			
		# compute velocity in Star Region
		u = 0.5*(self.u_L + self.u_R + f_R - f_L)
		
		return (p, u)
	
	def sample(self, p_star, u_star, s):
		"""
		Sample the solution, given the Star region pressure and velocity, 
		in terms of s = x/t
		
		TORO Fig 4.14
		s < u_star => left-hand side of the contact discontintuity
            p_L > p_star => left rarefication  => Toro (4.65)
                S < S_HL => initial state W_L
				S > S_HL
                    S > S_TL => star W_star_L, Toro (4.53)
					S < S_TL => inside left fan => Toro (4.56)
			p_L < p_star => left shock => Toro (4.64)
                S < S_L => initial state W_L
				S > S_L => star W_star_L, Toro (4.50)
		
		s > u_star => right-hand side
            p_R > p_star => right rarefication => Toro (4.67)
                S > S_HR => initial state W_R
				S < S_HR
                    S > S_TR => star W_star_R, Toro (4.60)
					S < S_TR => inside right fan => Toro (4.63)
			
			p_R < p_star => right shock => Toro (4.66)
                S > S_R => initial state W_R
				S < S_R => star W_star_R, Toro (4.57)
				
		
		"""
		if (s <= u_star): #to the left of the contact discontinuity 
			if (p_star <= self.p_L):
				# Left Rarefaction 
				a_star_L = self.a_L * (p_star/self.p_L) ** ((self.gamma-1)/(2*self.gamma)) # Toro (4.54)
				sHL = self.u_L - self.a_L # Toro (4.55)
				sTL = u_star - a_star_L

				if (s <= sHL): # initial state W_L
					rho = self.rho_L
					u  = self.u_L
					p   = self.p_L

				else:
					if (s > sTL): # star W_star_L
						rho = self.rho_L*(p_star/self.p_L)**(1.0 / self.gamma) # Toro (4.53)
						u  = u_star
						p   = p_star

					else: # point is inside left fan, Toro (4.56)
						rho = self.rho_L * (2/(self.gamma+1) + (self.gamma-1)*(self.u_L - s)/((self.gamma+1)*self.a_L))**(2 / (self.gamma - 1))
						u  = 2/(self.gamma+1)*(self.a_L + (self.gamma-1)/2*self.u_L + s)
						p   = self.p_L * (2/(self.gamma+1) + (self.gamma-1)*(self.u_L - s)/((self.gamma+1)*self.a_L))**(2*self.gamma/(self.gamma-1))
						c   = (2.0 / (self.gamma + 1.0))*(self.a_L + ((self.gamma - 1.0)/2.0)*(self.u_L - s)) # ???

			else:
				# Left shock 
				s_L = self.u_L - self.a_L*np.sqrt( (self.gamma+1)/(2*self.gamma) * (p_star/self.p_L) + ((self.gamma - 1) / (2*self.gamma))) # Toro (4.52)

				if (s <= s_L): # initial state W_L
					rho = self.rho_L
					u  = self.u_L
					p   = self.p_L

				else: # # star W_star_L
					rho = self.rho_L*((p_star/self.p_L) + (self.gamma-1.0)/(self.gamma+1.0))/((p_star/self.p_L)*(self.gamma-1.0)/(self.gamma+1) + 1.0) # Toro (4.50)
					u  = u_star
					p   = p_star

		else: # to the right of the contact discontinuity
			if (p_star > self.p_R):
				# Right Shock
				s_R = self.u_R + self.a_R*np.sqrt( (self.gamma+1.0)/(2.0*self.gamma) * (p_star/self.p_R) + ((self.gamma - 1.0) / (2.0*self.gamma))) # Toro (4.59)

				if (s >= s_R): # initial state W_R
					rho = self.rho_R
					u  = self.u_R
					p   = self.p_R

				else: # star W_star_R
					rho = self.rho_R*((p_star/self.p_R) + (self.gamma-1.0)/(self.gamma+1.0))/((p_star/self.p_R)*(self.gamma-1.0)/(self.gamma+1.0) + 1.0) # Toro (4.57)
					u  = u_star
					p   = p_star

			else:
				# Right Rarefaction
				a_star_R = self.a_R * (p_star/self.p_R) ** ((self.gamma-1.0)/(2.0*self.gamma)) # Toro (4.61)
				sHR = self.u_R + self.a_R # Toro (4.62)
				sTR = u_star + a_star_R

				if (s >= sHR): # initial W_R
					rho = self.rho_R
					u = self.u_R
					p = self.p_R

				else:
					if (s <= sTR):  # sar W_star_R
						rho = self.rho_R*(p_star/self.p_R)**(1.0/self.gamma) # Toro (4.60)
						u  = u_star
						p   = p_star

					else: # inside right fan, Toro (4.63)
						rho = self.rho_R * (2/(self.gamma+1.0) + (self.gamma-1.0)*(self.u_R - s)/((self.gamma+1.0)*self.a_R))**(2 / (self.gamma - 1.0))
						u  = 2/(self.gamma+1)*(-self.a_R + (self.gamma-1)/2*self.u_R + s)
						p   = self.p_R * (2/(self.gamma+1) + (self.gamma-1)*(self.u_R - s)/((self.gamma+1)*self.a_R))**(2*self.gamma/(self.gamma-1.0))
						c   = (2.0 / (self.gamma + 1.0))*(self.a_R + ((self.gamma - 1.0)/2.0)*(self.u_R - s)) # ???

		return (rho, u, p)
		
		
		
	def solve(self):
		
		# Check pressure positivity condition
		if ((2 / (self.gamma - 1))*(self.a_L+self.a_R) < (self.u_R - self.u_L)):
			print("Error: initial data is such that the vacuum is generated!")
			self.success = False
			
		# Find exact solution for pressure & velocity in star region
		(p_star, u_star) = self.calc_star_p_u()	
		
		for i in np.arange(len(self.x)):
			s = self.x[i]/self.t
			(rho, u, p) = self.sample(p_star, u_star, s)
			self.rho[i] = rho
			self.u[i]  = u
			self.p[i]   = p
			self.T[i] = p*28e-3/rho/8.31 # change according to EOS
			
		self.success = True
		return (self.x, self.rho, self.u, self.p, self.T)


"""
Equation of State: Ideal gas
"""
def EOS(p, rho, gamma): 
	e = p / (rho * (gamma-1))
	return e
def p_from_EOS(e, rho, gamma):
	p = e*rho*(gamma-1.0)
	return p

def primitive_to_conservative(rho, u, p, gamma):
	E = rho*EOS(p, rho, gamma) + 0.5 * rho * u**2
	return (rho, rho * u, E)

def conservative_to_primitive(q, gamma): # q is [rho, rho*u, E], E is p / (gamma - 1) + 0.5 * rho * u**2
	rho = q[0]
	u = q[1] / rho
	e = (q[2] - 0.5 * rho * u ** 2)/rho
	p = p_from_EOS(e, rho, gamma)
	T = p*28e-3/rho/8.31
	return (rho, u, p, T)

class HLL_RiemannSolver(): # according to Toro chapter 10

	def __init__(self, L, nx, CFL, q_L, q_R, gamma, t):

		self.L = L    # domain [-L/2,L/2], shock initially at x=0
		self.nx = nx   # resolution
		self.CFL = CFL   # Courant–Friedrichs–Lewy condition
		self.x = np.linspace(-self.L/2,self.L/2,self.nx)
		self.q = np.zeros([3, self.nx])
		self.set_init_state(q_L, q_R, gamma, t)


	def set_init_state(self, q_L, q_R, gamma, t):
		"""
		Set Left and Right Initial States
		"""

		q_L_array = np.array(q_L)
		q_R_array = np.array(q_R)
	
		self.q[:, :self.nx // 2] = np.repeat(q_L_array[:, np.newaxis], self.nx // 2, axis=1)
		self.q[:, self.nx // 2:] = np.repeat(q_R_array[:, np.newaxis], self.nx // 2, axis=1)

		self.gamma = gamma
		self.t_fin     = t
		
		rho_L, u_L, p_L, T_L = conservative_to_primitive(q_L, self.gamma)
		rho_R, u_R, p_R, T_R = conservative_to_primitive(q_R, self.gamma)
		
		self.success = False  # solve succesful?
		


	def flux(self, q): # q is [rho, rho*u, E]
		rho, u, p, T = conservative_to_primitive(q, self.gamma)
		E = q[2]
		return np.array([rho*u, rho*u*u + p, (E + p) * u])
	
	def Godunov_HLLC_flux(self, q_left, q_right):

		E_l = q_left[2]
		E_r = q_right[2]

		rho_l, u_l, p_l, T_l = conservative_to_primitive(q_left, self.gamma)
		rho_r, u_r, p_r, T_l = conservative_to_primitive(q_right, self.gamma)

		# sound speed
		aL = np.sqrt(self.gamma * p_l / rho_l)
		aR = np.sqrt(self.gamma * p_r / rho_r)

		# # 10.5.1 direct signal speed estimates
		# # Toro (10.48), the simples but _not recommended for practical computations_
		# Sl = np.min(u_l-aL, u_r-aR) 
		# Sr = np.max(u_l+aL, u_r+aR)

		#  10.5.2 Pressure–Based Wave Speed Estimates
		p_pvrs = 0.5*(p_l+p_r) - 0.5*(u_r-u_l)*0.5*(rho_l+rho_r)*0.5*(aL+aR) # Toro (10.67)
		p_star = np.max([0, p_pvrs])


		# 10.6 HLLC flux calculation
		# Pressure estimate
		# Toro (10.69)
		if (p_star > p_l):
			ql = np.sqrt(1.0+(self.gamma+1.0)*(p_star/p_l - 1.0)/(2*self.gamma))
		else:
			ql = 1

		if (p_star > p_r):
			qr = np.sqrt(1.0+(self.gamma+1.0)*(p_star/p_r - 1.0)/(2*self.gamma))
		else:
			qr = 1		
			
		# Wave speed estimate
		# Toro (10.68)
		Sl = u_l - aL*ql
		Sr = u_r + aR*qr

		# (10.70)
		S_star = (p_r -p_l + rho_l*u_l*(Sl-u_l)-rho_r*u_r*(Sr-u_r))/(rho_l*(Sl-u_l)-rho_r*(Sr-u_r))

		# HLLC flux

		F_l = self.flux(q_left)
		F_r = self.flux(q_right)

		# (10.73)
		q_star_l = np.multiply(rho_l * ((Sl - u_l)/(Sl - S_star)), [1, S_star, E_l/rho_l + (S_star - u_l)*(S_star + p_l/(rho_l*(Sl - u_l)))])
		q_star_r = np.multiply(rho_r * ((Sr - u_r)/(Sr - S_star)), [1, S_star, E_r/rho_r+ (S_star - u_r)*(S_star + p_r/(rho_r*(Sr - u_r)))])

		# (10.72), the same as (10.27) -- from  Rankine–Hugoniot Conditions for Sl, S_star, Sr
		F_star_l = F_l + Sl*(q_star_l - q_left)
		F_star_r = F_r + Sr*(q_star_r - q_right)

		# (10.71)
		if (0 <= Sl):
			F_hllc = F_l
		elif (Sl <= 0 and 0 <= S_star):
			F_hllc = F_star_l
		elif (S_star <= 0 and 0 <= Sr):
			F_hllc = F_star_r
		else:
			F_hllc = F_r

		return F_hllc
	
	def solve(self):
		"""
		Update the conserved variables U (density, momentum, energy) for each cell using the Godunov method.
		
		Parameters:
			q (ndarray): Array of conserved variables for each cell [rho, rho*u, E].
						Shape is (3, num_cells).
			hllc_flux (function): Function that calculates HLLC flux at an interface given left and right U vectors.
			dt (float): Time step size.
			dx (float): Spatial step size (cell width).
		
		Returns:
			q_new (ndarray): Updated conserved variables for each cell.
		"""
		dx = self.L / self.nx
		num_cells = self.nx

		rho, u, p, T_0 = conservative_to_primitive(self.q, self.gamma)
		print(rho)
		print(u)
		print(p)
		print(T_0)
		
		Times = [0]
		RHO = [rho]
		U   = [ u ]
		P   = [ p ]
		T = [T_0]
		print(f"Start calculating numerical solution for final time {self.t_fin}")
		while (Times[-1] < self.t_fin):
			q_new = np.copy(self.q)  # Copy the current state to store updates
			# Adaptive time step
			c = np.sqrt(self.gamma * p[0] / rho[0])
			dt = self.CFL * dx / c
			print(f"current T = {np.round(Times[-1], 4)}, with dt = {np.round(dt, 4)}")
	
			# Calculate fluxes at each interface

			fluxes = []
			for i in range(num_cells - 1):
				# Left and right states at each interface
				q_left = self.q[:, i]
				q_right = self.q[:, i + 1]
				
				# Calculate HLLC flux at the interface
				flux = self.Godunov_HLLC_flux(q_left, q_right)
				fluxes.append(flux)

			# Update each cell's conserved variables using the computed fluxes
			for i in range(1, num_cells - 1):
				# Flux difference across the cell (net flux)
				flux_in = fluxes[i - 1]
				flux_out = fluxes[i]
				
				# Update U based on net flux, scaled by dt/dx
				q_new[:, i] -= np.multiply((dt / dx), (flux_out - flux_in)) # (Toro 10.2)
			
			self.q = q_new
			rho_new, u_new, p_new, T_new = conservative_to_primitive(q_new, self.gamma)
			Times.append(Times[-1] + dt)
			RHO.append(rho_new)
			U.append(u_new)
			P.append(p_new)
			T.append(T_new)
			
		return (self.x, RHO, U, P, T, Times)



if __name__== "__main__":
	
	"""  Initial Conditions """
	
	# # Left State
	rho_L = 1.0
	u_L  = 0.0
	p_L   = 1.0

	# Right State
	rho_R = 0.125
	u_R  = 0.0
	p_R  = 0.1

	"""  Initial Conditions for chemistry """
	
    # # Left State
	# rho_L = 1.4
	# u_L  = 1000.0
	# p_L   = 1.3e5

	# # Right State
	# rho_R = 1.0
	# u_R  = 0.0
	# p_R  = 2e5

	# ideal gas gamma
	gamma = 5./3. 
	
    # domain properties
	L = 10.0    # domain [-L/2,L/2], shock initially at x=0
	nx = 500  # resolution
	CFL = 0.3   # Courant–Friedrichs–Lewy condition
		
	# final time 
	t = 1.0
    
	"""  NUMERICAL Riemann Solver """
	
	q_L = primitive_to_conservative(rho_L, u_L, p_L, gamma)
	q_R = primitive_to_conservative(rho_R, u_R, p_R, gamma)
	rs = HLL_RiemannSolver(L, nx, CFL, q_L, q_R, gamma, t)
	x, rho, u, p, T, times = rs.solve()
	N_times = len(times)
	print(f"Total timesteps number is {N_times}")
	# print(x)
	# print(np.shape(rho))
	# print(np.array(rho)[N_times-1, :])
	
	""" EXACT Riemann Solver """
	
	rho_e = [rho_L*(x<=0) + rho_R*(x>0)]
	u_e = [u_L*(x<=0) + u_R*(x>0)]
	p_e = [p_L*(x<=0) + p_R*(x>0)]
	T_e = [(p_L*28e-3/rho_L/8.31)*(x<=0) + (p_R*28e-3/rho_R/8.31)*(x>0)]
	# print(T_e)
	for t_i in times[1::]:
			# print(t_i)
			rs_e = ExactRiemannSolver(L, nx, CFL, rho_L, u_L, p_L, rho_R, u_R, p_R, gamma, t_i)
			x_i, rho_i, u_i, p_i, T_i = rs_e.solve()
			rho_e.append(rho_i)
			u_e.append(u_i)
			p_e.append(p_i)
			T_e.append(T_i)
	# print(np.shape(rho_e))
	# print(np.array(rho_e)[N_times-1, :])

	# """ Visualisation """
	
	# fig, (ax_rho, ax_u, ax_p) = plt.subplots(3, 1, figsize=(15, 10))
	# fig.suptitle('Time: ' + str(round(times[0], 3)), fontsize=12)
	# line_rho_init, = ax_rho.plot(x, rho_L*(x<=0) + rho_R*(x>0), color='k', label="Initial",linestyle="--", linewidth=0.7)
	# line_u_init, = ax_u.plot(x, u_L*(x<=0) + u_R*(x>0), color='k', label="Initial", linestyle="--", linewidth=0.7)
	# line_p_init, = ax_p.plot(x, p_L*(x<=0) + p_R*(x>0), color='k', label="initial", linestyle="--", linewidth=0.7)
	# line_rho, = ax_rho.plot([], [], color='b', label="Numerical")
	# line_u, = ax_u.plot([], [], color='r', label="Numerical")
	# line_p, = ax_p.plot([], [], color='g', label="Numerical")
	# line_rho_e, = ax_rho.plot([], [], color='b', label="Exact", linestyle="--", linewidth=0.7)
	# line_u_e, = ax_u.plot([], [], color='r', label="Exact", linestyle="--", linewidth=0.7)
	# line_p_e, = ax_p.plot([], [], color='g', label="Exact", linestyle="--", linewidth=0.7)
	# ax_rho.set_title("Density Evolution")
	# ax_rho.set_xlim(-L/2, L/2)
	# ax_rho.set_ylim(np.min(rho), 1.2*np.max(rho))
	# ax_rho.grid()
	# ax_rho.legend()
	# ax_rho.set_ylabel("rho")
	
	# ax_u.set_title("Velocity Evolution")
	# ax_u.set_xlim(-L/2, L/2)
	# ax_u.set_ylim(np.min(u), 1.2*np.max(u))
	# ax_u.grid()
	# ax_u.legend()
	# ax_u.set_ylabel("u")
	
	# ax_p.set_title("Pressure Evolution")
	# ax_p.set_xlim(-L/2, L/2)
	# ax_p.set_ylim(np.min(p), 1.2*np.max(p))
	# ax_p.grid()
	# ax_p.legend()
	# ax_p.set_ylabel("p")
	# ax_p.set_xlabel("x")

	
	# def update(frame):
	# 	fig.suptitle(f'Time: {round(times[frame], 3)}', fontsize=12)
	# 	line_rho.set_data(np.linspace(-L/2, L/2, nx), rho[frame])
	# 	line_u.set_data(np.linspace(-L/2, L/2, nx), u[frame])
	# 	line_p.set_data(np.linspace(-L/2, L/2, nx), p[frame])
	# 	# return line_rho, line_u, line_p
	# 	line_rho_e.set_data(np.linspace(-L/2, L/2, nx), rho_e[frame])
	# 	line_u_e.set_data(np.linspace(-L/2, L/2, nx), u_e[frame])
	# 	line_p_e.set_data(np.linspace(-L/2, L/2, nx), p_e[frame])
	# 	# return line_rho_e, line_u_e, line_p_e
	# 	return line_rho, line_u, line_p, line_rho_e, line_u_e, line_p_e
	
	# anim = FuncAnimation(fig, update, frames=len(times), blit=True, repeat=True, interval=50)
	# anim.save("Animation.gif")
	# plt.tight_layout()
	# plt.show()

	""" Visualisation (plot for T added) """
	
	fig, (ax_rho, ax_u, ax_p, ax_T) = plt.subplots(4, 1, figsize=(15, 10))
	fig.suptitle('Time: ' + str(round(times[0], 3)), fontsize=12)
	line_rho_init, = ax_rho.plot(x, rho_L*(x<=0) + rho_R*(x>0), color='k', label="Initial",linestyle="--", linewidth=0.7)
	line_u_init, = ax_u.plot(x, u_L*(x<=0) + u_R*(x>0), color='k', label="Initial", linestyle="--", linewidth=0.7)
	line_p_init, = ax_p.plot(x, p_L*(x<=0) + p_R*(x>0), color='k', label="initial", linestyle="--", linewidth=0.7)
	line_T_init, = ax_T.plot(x, p_L/rho_L*(x<=0) + p_R/rho_R*(x>0), color='k', label="initial", linestyle="--", linewidth=0.7)
	
	line_rho, = ax_rho.plot([], [], color='b', label="Numerical")
	line_u, = ax_u.plot([], [], color='r', label="Numerical")
	line_p, = ax_p.plot([], [], color='g', label="Numerical")
	line_T, = ax_T.plot([], [], color='r', label="Numerical")
	
	line_rho_e, = ax_rho.plot([], [], color='b', label="Exact", linestyle="--", linewidth=0.7)
	line_u_e, = ax_u.plot([], [], color='r', label="Exact", linestyle="--", linewidth=0.7)
	line_p_e, = ax_p.plot([], [], color='g', label="Exact", linestyle="--", linewidth=0.7)
	line_T_e, = ax_T.plot([], [], color='r', label="Exact", linestyle="--", linewidth=0.7)
	ax_rho.set_title("Density Evolution")
	ax_rho.set_xlim(-L/2, L/2)
	ax_rho.set_ylim(np.min(rho), 1.2*np.max(rho))
	ax_rho.grid()
	ax_rho.legend()
	ax_rho.set_ylabel("rho")
	
	ax_u.set_title("Velocity Evolution")
	ax_u.set_xlim(-L/2, L/2)
	ax_u.set_ylim(np.min(u), 1.2*np.max(u))
	ax_u.grid()
	ax_u.legend()
	ax_u.set_ylabel("u")
	
	ax_p.set_title("Pressure Evolution")
	ax_p.set_xlim(-L/2, L/2)
	ax_p.set_ylim(np.min(p), 1.2*np.max(p))
	ax_p.grid()
	ax_p.legend()
	ax_p.set_ylabel("p")

	ax_T.set_title("Temperature Evolution")
	ax_T.set_xlim(-L/2, L/2)
	ax_T.set_ylim(np.min(T), 1.2*np.max(T))
	ax_T.grid()
	ax_T.legend()
	ax_T.set_ylabel("T")
	ax_T.set_xlabel("x")

	
	def update(frame):
		fig.suptitle(f'Time: {round(times[frame], 3)}', fontsize=12)
		line_rho.set_data(np.linspace(-L/2, L/2, nx), rho[frame])
		line_u.set_data(np.linspace(-L/2, L/2, nx), u[frame])
		line_p.set_data(np.linspace(-L/2, L/2, nx), p[frame])
		line_T.set_data(np.linspace(-L/2, L/2, nx), T[frame])
		# return line_rho, line_u, line_p
		line_rho_e.set_data(np.linspace(-L/2, L/2, nx), rho_e[frame])
		line_u_e.set_data(np.linspace(-L/2, L/2, nx), u_e[frame])
		line_p_e.set_data(np.linspace(-L/2, L/2, nx), p_e[frame])
		line_T_e.set_data(np.linspace(-L/2, L/2, nx), T_e[frame])
		# return line_rho_e, line_u_e, line_p_e
		return line_rho, line_u, line_p, line_T, line_rho_e, line_u_e, line_p_e, line_T_e
	
	anim = FuncAnimation(fig, update, frames=len(times), blit=True, repeat=True, interval=100)
	anim.save("Animation.gif")
	plt.tight_layout()
	plt.show()

	


	
	


	