// Mock BetterAuth library for demonstration
const BetterAuth = {
  signUp: async (email, password, backgroundInfo) => {
    console.log('BetterAuth: Signing up...', {email, password, backgroundInfo}); // Added password
    // Simulate API call
    return new Promise((resolve) => {
      setTimeout(() => {
        let users = JSON.parse(localStorage.getItem('users') || '[]');
        console.log('Users before signup:', users); // Log users array before signup

        if (users.find(user => user.email === email)) {
          console.log('User already exists:', email); // Log if user exists
          resolve({success: false, message: 'User already exists'});
        } else {
          users.push({email, password, backgroundInfo, username: email.split('@')[0]});
          localStorage.setItem('users', JSON.stringify(users));
          localStorage.setItem('currentUser', JSON.stringify({email, username: email.split('@')[0]}));
          console.log('Users after signup:', users); // Log users array after signup
          resolve({success: true});
        }
      }, 1000);
    });
  },
  signIn: async (email, password) => {
    console.log('BetterAuth: Signing in...', {email, password}); // Added password
    // Simulate API call
    return new Promise((resolve) => {
      setTimeout(() => {
        const users = JSON.parse(localStorage.getItem('users') || '[]');
        console.log('Users during signin check:', users); // Log users during signin check
        const user = users.find(u => u.email === email && u.password === password);
        if (user) {
          localStorage.setItem('currentUser', JSON.stringify({email, username: user.username}));
          console.log('User signed in:', email); // Log user signed in
          resolve({success: true});
        } else {
          console.log('Invalid credentials for:', email); // Log invalid credentials
          resolve({success: false, message: 'Invalid credentials'});
        }
      }, 1000);
    });
  },
  getCurrentUser: () => {
    const user = localStorage.getItem('currentUser');
    console.log('Current user from localStorage:', user); // Log current user
    return user ? JSON.parse(user) : null;
  },
  signOut: () => {
    console.log('Signing out user:', BetterAuth.getCurrentUser()?.email); // Log user signing out
    localStorage.removeItem('currentUser');
  },
};

export default BetterAuth;