module mod_buffer
    use mod_random, only: randn
    use mod_network, only: network_type
    implicit none
    
    private
    public :: buffer_type
    
    type :: buffer_type
        integer :: buffer_capacity, batch_size, buffer_counter
        real, allocatable :: state_buffer(:,:), action_buffer(:,:), reward_buffer(:), next_state_buffer(:,:)
    contains
        procedure, public, pass(self) :: init
        procedure, public, pass(self) :: nrecord 
        procedure, public, pass(self) :: update
        procedure, public, pass(self) :: learn
    end type buffer_type
    
interface buffer_type
    module procedure :: buffer_constructor
end interface buffer_type    

contains
    
    type(buffer_type) function buffer_constructor(buffer_capacity, batch_size, num_states, num_actions) result(buffer)
        implicit none
        integer, intent(in) :: buffer_capacity, batch_size, num_states, num_actions
        call buffer%init(buffer_capacity, batch_size, num_states, num_actions)
    end function buffer_constructor
    
    subroutine init(self, buffer_capacity, batch_size, num_states, num_actions)
        implicit none
        class(buffer_type), intent(in out) :: self
        integer, intent(in) :: buffer_capacity, batch_size, num_states, num_actions
        allocate(self%state_buffer(num_states,0:buffer_capacity-1))
        allocate(self%action_buffer(num_actions,0:buffer_capacity-1))
        allocate(self%reward_buffer(0:buffer_capacity-1))
        allocate(self%next_state_buffer(num_states,0:buffer_capacity-1))
        ! Number of "experiences" to store at max
        self%buffer_capacity = buffer_capacity
        ! Num of tuples to train on.
        self%batch_size = batch_size
        
        ! Its tells us num of times record() was called.
        self%buffer_counter = 0
        
        ! Instead of list of tuples as the exp.replay concept go
        ! We use different structs for each tuple element
        self%state_buffer = 0   
        self%action_buffer = 0
        self%reward_buffer = 0
        self%next_state_buffer = 0
    end subroutine init
    
    ! Takes (s,a,r,s') column obervation tuple as input
    subroutine nrecord(self, prev_state, action, reward, state)
        implicit none
        class(buffer_type), intent(in out) :: self
        integer :: index
        real, intent(in) :: prev_state(:), state(:), action(:)
        real, intent(in) :: reward
        
        ! Set index to zero if buffer_capacity is exceeded,
        ! replacing old records
        index = mod(self%buffer_counter, self%buffer_capacity)

        self%state_buffer(:,index) = prev_state
        self%action_buffer(:,index) = action
        self%reward_buffer(index) = reward
        self%next_state_buffer(:,index) = state

        self%buffer_counter = self%buffer_counter + 1
    end subroutine nrecord
    
    ! Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    ! TensorFlow to build a static graph out of the logic and computations in our function.
    ! This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    !@tf.function
    subroutine update(self, state_batch, action_batch, reward_batch, next_state_batch, actor_model, critic_model_1, critic_model_2, critic_model, target_actor, target_critic_1, target_critic_2, target_critic, critic_lr, actor_lr, gamma)
        ! Training and updating Actor & Critic networks.
        ! See Pseudo Code.
        implicit none
        class(buffer_type), intent(in out) :: self
        type(network_type) :: actor_model, critic_model_1, critic_model_2, critic_model, target_actor, target_critic_1, target_critic_2, target_critic
        real, intent(in) :: state_batch(:,:), action_batch(:,:), next_state_batch(:,:)
        real, intent(in) :: reward_batch(:)
        real, intent(in) :: critic_lr, actor_lr, gamma
        real :: target_actions(size(action_batch, dim=1),size(action_batch, dim=2)), critic_value(1,size(action_batch, dim=2)), actions(size(action_batch, dim=1),size(action_batch, dim=2))
        real :: critic_value_1(32,size(action_batch, dim=2)), critic_value_2(32,size(action_batch, dim=2))
        real :: y1(32,size(reward_batch)), y2(32,size(reward_batch))
        real :: y12(64,size(reward_batch)), critic_value_12(64,size(action_batch, dim=2))
        real :: y(1,size(reward_batch))
        real :: reward_batch_matrix(1,size(reward_batch))
        integer :: i
        
        reward_batch_matrix(1,:) = reward_batch(:)
        
        target_actions = target_actor%output(next_state_batch) !!
        y1 = target_critic_1%output(next_state_batch)
        y2 = target_critic_2%output(target_actions)
        !!此处需要想想数据结构,如何把两个输入拼接,(Fortran貌似不会智能矩阵拼接,尝试手动拼接)
        do i = 1, size(reward_batch)
            y12(:,i) = [y1(:,i), y2(:,i)]
        end do
        !print *, shape(y12)
        y = reward_batch_matrix + gamma * target_critic%output(y12)
        !print *, y(1,1)
        critic_value_1 = critic_model_1%output(state_batch)
        critic_value_2 = critic_model_2%output(action_batch)
        do i = 1, size(action_batch, dim=2)
            critic_value_12(:,i) = [critic_value_1(:,i), critic_value_2(:,i)]
        end do
        critic_value = critic_model%output(critic_value_12)
        call critic_model_1%train(state_batch, y1, critic_lr)
        !print *, state_batch(1,1:2)
        !print *, reward_batch(1:2)
        call critic_model_2%train(action_batch, y2, critic_lr)
        call critic_model%train(critic_value_12, y, critic_lr)
        !print *, critic_value_12(1,1) --需要检查为什么不变化
        !print *, critic_model%layers(1)%w(1,1) --需要检查为什么不变化
        actions = actor_model%output(state_batch)
        
        !y(1,:) = 10
        
        critic_value_1 = critic_model_1%output(state_batch)
        critic_value_2 = critic_model_2%output(actions)
        do i = 1, size(action_batch, dim=2)
            critic_value_12(:,i) = [critic_value_1(:,i), critic_value_2(:,i)]
        end do
        critic_value = critic_model%output(critic_value_12)
        ! Used `-value` as we want to maximize the value given
        ! by the critic for our actions
        !call actor_model%train(critic_value, 0, eta)
        !actor_loss = -(sum(critic_value) / size(critic_value)) !!这一步可能需要自己重写反向传播算法
        !print *, actor_model%loss(critic_value(:,1), y(:,1))
        !print *, -critic_value(:,1)
        
        call actor_model%train_maximize_batch(state_batch, actor_lr)
        !call actor_model%train_maximize_batch(state_batch, actor_lr)
    end subroutine update
    
    ! We compute the loss and update parameters
    subroutine learn(self, actor_model, critic_model_1, critic_model_2, critic_model, target_actor, target_critic_1, target_critic_2, target_critic, critic_lr, actor_lr, gamma)
        implicit none
        class(buffer_type), intent(in out) :: self
        real, intent(in) :: critic_lr, actor_lr, gamma
        type(network_type) :: actor_model, critic_model_1, critic_model_2, critic_model, target_actor, target_critic_1, target_critic_2, target_critic
        real :: k(self%batch_size)
        real :: state_batch(size(self%state_buffer, dim=1), self%batch_size), action_batch(size(self%action_buffer, dim=1), self%batch_size)
        real :: reward_batch(self%batch_size), next_state_batch(size(self%next_state_buffer, dim=1), self%batch_size)
        integer :: batch_indices(self%batch_size)
        integer :: record_range
        
        ! Get sampling range
        record_range = min(self%buffer_counter, self%buffer_capacity)
        ! Randomly sample indices
        !call random_number(record_range, self%batch_size)
        call random_seed()
        call random_number(k)
        k = k * record_range
        batch_indices = floor(k)

        ! Pick corresponding experiences
        state_batch = self%state_buffer(:,batch_indices)
        action_batch = self%action_buffer(:,batch_indices)
        reward_batch = self%reward_buffer(batch_indices)
        next_state_batch = self%next_state_buffer(:,batch_indices)

        call self%update(state_batch, action_batch, reward_batch, next_state_batch, actor_model, critic_model_1, critic_model_2, critic_model, target_actor, target_critic_1, target_critic_2, target_critic, critic_lr, actor_lr, gamma)
    end subroutine learn
end module mod_buffer