module mod_OUActionNoise
    use mod_random, only: randn
    implicit none
    
    private
    public :: noise_type
    
    type :: noise_type
        real :: mean, std_dev, theta, dt, x_prev
    contains
        procedure, public, pass(self) :: init, ncall
    end type noise_type
    
interface noise_type
    module procedure :: noise_constructor
end interface noise_type    

contains   
    type(noise_type) function noise_constructor(mean, std_deviation) result(ou_noise)
        implicit none
        real, intent(in) :: mean, std_deviation
        real :: theta=0.15, dt=0.01, x_prev = 0
        call ou_noise%init(mean, std_deviation, theta, dt, x_prev)
    end function noise_constructor
    
    subroutine init(self, mean, std_deviation, theta, dt, x_prev)
        implicit none
        class(noise_type), intent(in out) :: self
        real, intent(in) :: mean, std_deviation, theta, dt
        real :: x_prev
        self%theta = theta
        self%mean = mean
        self%std_dev = std_deviation
        self%dt = dt
        self%x_prev = x_prev
    end subroutine init
    
    real function ncall(self) result(x)
        implicit none
        class(noise_type), intent(in out) :: self
        real :: rand(1)
        rand = randn(1)
        x = self%x_prev + self%theta * (self%mean - self%x_prev) * self%dt + self%std_dev * sqrt(self%dt) * rand(1)
        ! Store x into x_prev
        ! Makes next noise dependent on current one
        self%x_prev = x
        !print *, x, self%x_prev, self%theta, mean, self%dt, std_deviation
    end function ncall
end module mod_OUActionNoise
    
!program test
!    use mod_kinds, only: rk, ik
!    use mod_OUActionNoise, only: noise_type
!    use mod_network, only: network_type
!    implicit none
!    type(network_type) :: net
!    type(noise_type) :: ou_noise
!    real(rk) :: noise, mean, std_dev
!    integer :: i
!    mean = 0
!    std_dev = 0.2
!    ou_noise = noise_type(mean, std_dev)
!    do i = 1, 50
!       noise = ou_noise%ncall()
!        print *, ou_noise
!        print *, noise
!    end do
!    pause
!    stop
!end program test