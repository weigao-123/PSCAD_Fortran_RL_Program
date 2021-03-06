!=======================================================================
! Generated by  : PSCAD v4.6.3.0
!
! Warning:  The content of this file is automatically generated.
!           Do not modify, as any changes made here will be lost!
!-----------------------------------------------------------------------
! Component     : Main
! Description   : 
!-----------------------------------------------------------------------


!=======================================================================

      SUBROUTINE MainDyn()

!---------------------------------------
! Standard includes
!---------------------------------------

      INCLUDE 'nd.h'
      INCLUDE 'emtconst.h'
      INCLUDE 'emtstor.h'
      INCLUDE 's0.h'
      INCLUDE 's1.h'
      INCLUDE 's2.h'
      INCLUDE 's4.h'
      INCLUDE 'branches.h'
      INCLUDE 'pscadv3.h'
      INCLUDE 'fnames.h'
      INCLUDE 'radiolinks.h'
      INCLUDE 'matlab.h'
      INCLUDE 'rtconfig.h'

!---------------------------------------
! Function/Subroutine Declarations
!---------------------------------------

!     SUBR    EMTDC_X2COMP  ! 'Comparator with Interpolation'
      REAL    REALPOLE      ! Real Pole

!---------------------------------------
! Variable Declarations
!---------------------------------------


! Subroutine Arguments

! Electrical Node Indices

! Control Signals
      INTEGER  Simulation_step, IT_1, IT_2
      REAL     RT_1, RT_2, RT_3, RT_4, RT_5, RT_6
      REAL     RT_7, RT_8, RT_9, RT_10, RT_11
      REAL     RT_12, RT_13, clk

! Internal Variables
      REAL     RVD2_1(2), state(2), action(2)

! Indexing variables
      INTEGER ICALL_NO                            ! Module call num
      INTEGER ISTOI, ISTOF, IT_0                  ! Storage Indices
      INTEGER ICX, IPGB                           ! Control/Monitoring
      INTEGER SS                                  ! SS/Node/Branch/Xfmr


!---------------------------------------
! Local Indices
!---------------------------------------

! Dsdyn <-> Dsout transfer index storage

      NTXFR = NTXFR + 1

      TXFR(NTXFR,1) = NSTOL
      TXFR(NTXFR,2) = NSTOI
      TXFR(NTXFR,3) = NSTOF
      TXFR(NTXFR,4) = NSTOC

! Define electric network subsystem number

      SS        = NODE(NNODE+1)

! Increment and assign runtime configuration call indices

      ICALL_NO  = NCALL_NO
      NCALL_NO  = NCALL_NO + 1

! Increment global storage indices

      ISTOI     = NSTOI
      NSTOI     = NSTOI + 3
      ISTOF     = NSTOF
      NSTOF     = NSTOF + 14
      IPGB      = NPGB
      NPGB      = NPGB + 3
      ICX       = NCX
      NCX       = NCX + 1
      NNODE     = NNODE + 2
      NCSCS     = NCSCS + 0
      NCSCR     = NCSCR + 0

!---------------------------------------
! Transfers from storage arrays
!---------------------------------------

      RT_1     = STOF(ISTOF + 1)
      RT_2     = STOF(ISTOF + 2)
      RT_3     = STOF(ISTOF + 3)
      RT_4     = STOF(ISTOF + 4)
      RT_5     = STOF(ISTOF + 5)
      RT_6     = STOF(ISTOF + 6)
      RT_7     = STOF(ISTOF + 7)
      RT_8     = STOF(ISTOF + 8)
      Simulation_step = STOI(ISTOI + 1)
      RT_9     = STOF(ISTOF + 9)
      RT_10    = STOF(ISTOF + 10)
      RT_11    = STOF(ISTOF + 11)
      RT_12    = STOF(ISTOF + 12)
      IT_1     = STOI(ISTOI + 2)
      IT_2     = STOI(ISTOI + 3)
      RT_13    = STOF(ISTOF + 13)
      clk      = STOF(ISTOF + 14)


!---------------------------------------
! Electrical Node Lookup
!---------------------------------------


!---------------------------------------
! Configuration of Models
!---------------------------------------

      IF ( TIMEZERO ) THEN
         FILENAME = 'Main.dta'
         CALL EMTDC_OPENFILE
         SECTION = 'DATADSD:'
         CALL EMTDC_GOTOSECTION
      ENDIF
!---------------------------------------
! Generated code from module definition
!---------------------------------------


! 10:[const] Real Constant 

      RT_8 = 0.0

! 20:[const] Real Constant 

      RT_11 = 30.0

! 30:[time-sig] Output of Simulation Time 
      RT_10 = TIME

! 40:[compare] Single Input Level Comparator 
!
!
      CALL EMTDC_X2COMP(0,0,0.25,RT_10,60.0,0.0,30.0,RVD2_1)
      RT_1 = RVD2_1(1)

! 50:[impulse] Impulse Generator /w Interpolation 
      CALL E_XIGEN1_EXE(RVD2_1)
      RT_13 = RVD2_1(1)
!

! 60:[consti] Integer Constant 

      IT_1 = 0

! 70:[var_switch] Two State Switch 'DDPG'
      IT_2 = NINT(CX(CXMAP(ICX+1)))

! 80:[select] Two Input Selector 
      IF (IT_2 .EQ. RTCI(NRTCI)) THEN
         clk = RT_13
      ELSE
         clk = REAL(IT_1)
      ENDIF
      NRTCI = NRTCI + 1
!

! 90:[gain] Gain Block 
!  Gain
      RT_6 = -10.0 * RT_7

! 100:[sumjct] Summing/Differencing Junctions 
      RT_3 = + RT_1 - RT_2

! 110:[abs] Absolute Value 
!
      RT_7 = ABS(RT_3)
!

! 120:[compar] Two Input Comparator 
!
      CALL EMTDC_X2COMP(0,0,RT_7,RT_11,-100.0,0.0,0.0,RVD2_1)
      RT_12 = RVD2_1(1)

! 130:[sumjct] Summing/Differencing Junctions 
      RT_9 = + RT_6 + RT_12

! 140:[RL_Agent_DDPG]  'ddpg_test'
        state(1) = RT_3
        action(1) = RT_4
        IF(clk.GT.0.9) THEN
                call ddpg(RT_3,RT_9,RT_8,Simulation_step,RT_4,Simulation&
     &_step)
        ENDIF

! 150:[realpole] Real Pole 
!  Real_Pole
      RT_5 = REALPOLE(0,1,0,5.0,0.1,RT_4,0.0,-1.0E20,1.0E20)

! 160:[realpole] Real Pole 
!  Real_Pole
      RT_2 = REALPOLE(0,1,0,6.0,0.01,RT_5,0.0,-1.0E20,1.0E20)

! 170:[pgb] Output Channel 'Reward'

      PGB(IPGB+1) = RT_9

! 180:[pgb] Output Channel 'Out'

      PGB(IPGB+2) = RT_2

! 190:[pgb] Output Channel 'Act'

      PGB(IPGB+3) = RT_4

!---------------------------------------
! Feedbacks and transfers to storage
!---------------------------------------

      STOF(ISTOF + 1) = RT_1
      STOF(ISTOF + 2) = RT_2
      STOF(ISTOF + 3) = RT_3
      STOF(ISTOF + 4) = RT_4
      STOF(ISTOF + 5) = RT_5
      STOF(ISTOF + 6) = RT_6
      STOF(ISTOF + 7) = RT_7
      STOF(ISTOF + 8) = RT_8
      STOI(ISTOI + 1) = Simulation_step
      STOF(ISTOF + 9) = RT_9
      STOF(ISTOF + 10) = RT_10
      STOF(ISTOF + 11) = RT_11
      STOF(ISTOF + 12) = RT_12
      STOI(ISTOI + 2) = IT_1
      STOI(ISTOI + 3) = IT_2
      STOF(ISTOF + 13) = RT_13
      STOF(ISTOF + 14) = clk


!---------------------------------------
! Transfer to Exports
!---------------------------------------

!---------------------------------------
! Close Model Data read
!---------------------------------------

      IF ( TIMEZERO ) CALL EMTDC_CLOSEFILE
      RETURN
      END

!=======================================================================

      SUBROUTINE MainOut()

!---------------------------------------
! Standard includes
!---------------------------------------

      INCLUDE 'nd.h'
      INCLUDE 'emtconst.h'
      INCLUDE 'emtstor.h'
      INCLUDE 's0.h'
      INCLUDE 's1.h'
      INCLUDE 's2.h'
      INCLUDE 's4.h'
      INCLUDE 'branches.h'
      INCLUDE 'pscadv3.h'
      INCLUDE 'fnames.h'
      INCLUDE 'radiolinks.h'
      INCLUDE 'matlab.h'
      INCLUDE 'rtconfig.h'

!---------------------------------------
! Function/Subroutine Declarations
!---------------------------------------


!---------------------------------------
! Variable Declarations
!---------------------------------------


! Electrical Node Indices

! Control Signals
      INTEGER  IT_1
      REAL     RT_8, RT_11

! Internal Variables

! Indexing variables
      INTEGER ICALL_NO                            ! Module call num
      INTEGER ISTOL, ISTOI, ISTOF, ISTOC          ! Storage Indices
      INTEGER SS                                  ! SS/Node/Branch/Xfmr


!---------------------------------------
! Local Indices
!---------------------------------------

! Dsdyn <-> Dsout transfer index storage

      NTXFR = NTXFR + 1

      ISTOL = TXFR(NTXFR,1)
      ISTOI = TXFR(NTXFR,2)
      ISTOF = TXFR(NTXFR,3)
      ISTOC = TXFR(NTXFR,4)

! Define electric network subsystem number

      SS        = NODE(NNODE+1)

! Increment and assign runtime configuration call indices

      ICALL_NO  = NCALL_NO
      NCALL_NO  = NCALL_NO + 1

! Increment global storage indices

      NPGB      = NPGB + 3
      NCX       = NCX + 0
      NNODE     = NNODE + 2
      NCSCS     = NCSCS + 0
      NCSCR     = NCSCR + 0

!---------------------------------------
! Transfers from storage arrays
!---------------------------------------

      RT_8     = STOF(ISTOF + 8)
      RT_11    = STOF(ISTOF + 11)
      IT_1     = STOI(ISTOI + 2)


!---------------------------------------
! Electrical Node Lookup
!---------------------------------------


!---------------------------------------
! Configuration of Models
!---------------------------------------

      IF ( TIMEZERO ) THEN
         FILENAME = 'Main.dta'
         CALL EMTDC_OPENFILE
         SECTION = 'DATADSO:'
         CALL EMTDC_GOTOSECTION
      ENDIF
!---------------------------------------
! Generated code from module definition
!---------------------------------------


! 10:[const] Real Constant 

      RT_8 = 0.0

! 20:[const] Real Constant 

      RT_11 = 30.0

! 60:[consti] Integer Constant 

      IT_1 = 0

!---------------------------------------
! Feedbacks and transfers to storage
!---------------------------------------

      STOF(ISTOF + 8) = RT_8
      STOF(ISTOF + 11) = RT_11
      STOI(ISTOI + 2) = IT_1


!---------------------------------------
! Close Model Data read
!---------------------------------------

      IF ( TIMEZERO ) CALL EMTDC_CLOSEFILE
      RETURN
      END

!=======================================================================

      SUBROUTINE MainDyn_Begin()

!---------------------------------------
! Standard includes
!---------------------------------------

      INCLUDE 'nd.h'
      INCLUDE 'emtconst.h'
      INCLUDE 's0.h'
      INCLUDE 's1.h'
      INCLUDE 's4.h'
      INCLUDE 'branches.h'
      INCLUDE 'pscadv3.h'
      INCLUDE 'radiolinks.h'
      INCLUDE 'rtconfig.h'

!---------------------------------------
! Function/Subroutine Declarations
!---------------------------------------


!---------------------------------------
! Variable Declarations
!---------------------------------------


! Subroutine Arguments

! Electrical Node Indices

! Control Signals
      INTEGER  IT_1
      REAL     RT_8, RT_11

! Internal Variables

! Indexing variables
      INTEGER ICALL_NO                            ! Module call num
      INTEGER ICX                                 ! Control/Monitoring
      INTEGER SS                                  ! SS/Node/Branch/Xfmr


!---------------------------------------
! Local Indices
!---------------------------------------


! Define electric network subsystem number

      SS        = NODE(NNODE+1)

! Increment and assign runtime configuration call indices

      ICALL_NO  = NCALL_NO
      NCALL_NO  = NCALL_NO + 1

! Increment global storage indices

      ICX       = NCX
      NCX       = NCX + 1
      NNODE     = NNODE + 2
      NCSCS     = NCSCS + 0
      NCSCR     = NCSCR + 0

!---------------------------------------
! Electrical Node Lookup
!---------------------------------------


!---------------------------------------
! Generated code from module definition
!---------------------------------------


! 10:[const] Real Constant 
      RT_8 = 0.0

! 20:[const] Real Constant 
      RT_11 = 30.0

! 30:[time-sig] Output of Simulation Time 

! 40:[compare] Single Input Level Comparator 

! 50:[impulse] Impulse Generator /w Interpolation 
      CALL COMPONENT_ID(ICALL_NO,861787779)
      CALL E_XIGEN1_CFG(0,0.0,1.0,1000.0)

! 60:[consti] Integer Constant 
      IT_1 = 0

! 70:[var_switch] Two State Switch 'DDPG'

! 80:[select] Two Input Selector 
      RTCI(NRTCI) = 1
      NRTCI = NRTCI + 1

! 90:[gain] Gain Block 

! 100:[sumjct] Summing/Differencing Junctions 

! 110:[abs] Absolute Value 

! 120:[compar] Two Input Comparator 

! 130:[sumjct] Summing/Differencing Junctions 

! 140:[RL_Agent_DDPG]  'ddpg_test'

! 150:[realpole] Real Pole 

! 160:[realpole] Real Pole 

! 170:[pgb] Output Channel 'Reward'

! 180:[pgb] Output Channel 'Out'

! 190:[pgb] Output Channel 'Act'

      RETURN
      END

!=======================================================================

      SUBROUTINE MainOut_Begin()

!---------------------------------------
! Standard includes
!---------------------------------------

      INCLUDE 'nd.h'
      INCLUDE 'emtconst.h'
      INCLUDE 's0.h'
      INCLUDE 's1.h'
      INCLUDE 's4.h'
      INCLUDE 'branches.h'
      INCLUDE 'pscadv3.h'
      INCLUDE 'radiolinks.h'
      INCLUDE 'rtconfig.h'

!---------------------------------------
! Function/Subroutine Declarations
!---------------------------------------


!---------------------------------------
! Variable Declarations
!---------------------------------------


! Subroutine Arguments

! Electrical Node Indices

! Control Signals
      INTEGER  IT_1
      REAL     RT_8, RT_11

! Internal Variables

! Indexing variables
      INTEGER ICALL_NO                            ! Module call num
      INTEGER SS                                  ! SS/Node/Branch/Xfmr


!---------------------------------------
! Local Indices
!---------------------------------------


! Define electric network subsystem number

      SS        = NODE(NNODE+1)

! Increment and assign runtime configuration call indices

      ICALL_NO  = NCALL_NO
      NCALL_NO  = NCALL_NO + 1

! Increment global storage indices

      NCX       = NCX + 0
      NNODE     = NNODE + 2
      NCSCS     = NCSCS + 0
      NCSCR     = NCSCR + 0

!---------------------------------------
! Electrical Node Lookup
!---------------------------------------


!---------------------------------------
! Generated code from module definition
!---------------------------------------


! 10:[const] Real Constant 
      RT_8 = 0.0

! 20:[const] Real Constant 
      RT_11 = 30.0

! 60:[consti] Integer Constant 
      IT_1 = 0

      RETURN
      END

