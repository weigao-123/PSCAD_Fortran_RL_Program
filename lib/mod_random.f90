module mod_random

  ! Provides a random number generator with
  ! normal distribution, centered on zero.

  implicit none

  private
  public :: randn

  real, parameter :: pi = 3.1415926

  interface randn
    module procedure :: randn1d, randn2d
  end interface randn

contains

  function randn1d(n) result(r)
    ! Generates n random numbers with a normal distribution.
    integer, intent(in) :: n
    real :: r(n), r2(n)
    call random_number(r)
    call random_number(r2)
    r = sqrt(-2 * log(r)) * cos(2 * pi * r2)
  end function randn1d

  function randn2d(m, n) result(r)
    ! Generates m x n random numbers with a normal distribution.
    integer, intent(in) :: m, n
    real :: r(m, n), r2(m, n)
    call random_number(r)
    call random_number(r2)
    r = sqrt(-2 * log(r)) * cos(2 * pi * r2)
  end function randn2d

end module mod_random
