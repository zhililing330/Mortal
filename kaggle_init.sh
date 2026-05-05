#!/bin/bash

# --- 配置区 ---
# 1. 你的私有/分支仓库地址
REPO_URL="https://github.com/zhililing330/Mortal.git"
# 2. 你之前备份到网盘或直接准备上传的 200M 权重/环境压缩包文件名
BACKUP_ZIP="mortal_compiled_core.zip"
# 3. 设置 SSH 密码（用于 Remote-SSH 连接）
USER_PASSWORD="zjp@12345"

echo "🚀 开始初始化 Kaggle 训练环境..."

# --- 第一步：系统依赖与 SSH 恢复 ---
echo "📦 安装系统组件..."
apt-get update && apt-get install -y openssh-server zip unzip rsync
echo "root:$USER_PASSWORD" | chpasswd
mkdir -p /var/run/sshd
# 允许密码登录
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
service ssh start

# --- 第二步：代码恢复 ---
echo "git 拉取项目代码..."
cd /content
if [ -d "Mortal" ]; then
    echo "项目文件夹已存在，执行拉取更新..."
    cd Mortal && git pull
else
    git clone $REPO_URL
    cd Mortal
fi

# --- 第三步：恢复编译成果与权重 ---
# 逻辑：如果你已经通过 Cursor 把备份 ZIP 传到了 /content 目录下
if [ -f "/content/$BACKUP_ZIP" ]; then
    echo "找到环境/权重备份，正在解压..."
    unzip -o "/content/$BACKUP_ZIP" -d ./
    echo "✅ 编译成品与权重已恢复。"
else
    echo "⚠️ 未发现备份文件 $BACKUP_ZIP，请手动上传或检查路径。"
fi

# --- 第四步：Python 环境恢复 ---
echo "🐍 安装 Python 依赖 (Mortal 专用)..."
pip install -r requirements.txt
pip install kanmuri

# --- 第五步：验证运行 ---
echo "🔍 检查运行状态..."
if [ -f "./target/release/mortal" ]; then
    echo "✅ 引擎二进制文件就绪！"
    ./target/release/mortal --version
else
    echo "❌ 引擎文件缺失，可能需要重新编译或检查备份。"
fi

echo "✨ 初始化完成！你现在可以连接 Remote-SSH 并开始训练了。"
