"""
生成密码哈希的工具脚本
运行这个脚本来生成新的密码哈希，然后替换到 config.yaml 中
"""

import streamlit_authenticator as stauth

def generate_password_hashes():
    """生成常用密码的哈希值"""
    
    passwords_to_hash = {
        'admin123': None,
        'research123': None, 
        'user123': None
    }
    
    # 生成哈希
    hashed_passwords = stauth.Hasher(list(passwords_to_hash.keys())).generate()
    
    print("生成的密码哈希值：")
    print("=" * 50)
    
    for i, password in enumerate(passwords_to_hash.keys()):
        passwords_to_hash[password] = hashed_passwords[i]
        print(f"原密码: {password}")
        print(f"哈希值: {hashed_passwords[i]}")
        print("-" * 30)
    
    return passwords_to_hash

def generate_custom_password(password):
    """为自定义密码生成哈希"""
    hashed = stauth.Hasher([password]).generate()
    print(f"密码 '{password}' 的哈希值: {hashed[0]}")
    return hashed[0]

if __name__ == "__main__":
    print("Sci-Flow 密码哈希生成器")
    print("=" * 40)
    
    # 生成预设密码的哈希
    generate_password_hashes()
    
    # 可以生成自定义密码
    print("\n如果需要生成自定义密码的哈希，请修改下面的代码：")
    # 取消注释下面这行来生成自定义密码
    # generate_custom_password("your_custom_password")