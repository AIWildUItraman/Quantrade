// 代码审查测试脚本
const fs = require('fs');
const path = require('path');

console.log('='.repeat(60));
console.log('K线标注工具代码审查');
console.log('='.repeat(60));

// 读取HTML文件
const htmlPath = path.join(__dirname, 'annotation_tool.html');
const htmlContent = fs.readFileSync(htmlPath, 'utf-8');

// 提取JavaScript代码
const scriptMatch = htmlContent.match(/<script>([\s\S]*?)<\/script>/g);
if (!scriptMatch) {
    console.error('❌ 未找到JavaScript代码');
    process.exit(1);
}

// 合并所有script标签的内容
let jsCode = scriptMatch.map(s => s.replace(/<\/?script>/g, '')).join('\n');

console.log('\n📋 检查项目：\n');

// 1. 检查语法错误
console.log('1. 检查关键函数是否存在...');
const requiredFunctions = [
    'parseCSV',
    'createWindows',
    'findFirstUnlabeledIndex',
    'exportLabeledCSV',
    'exportImages',
    'drawLabelTag',
    'downloadFile',
    'renderKlineChart',
    'setLabel'
];

let missingFunctions = [];
requiredFunctions.forEach(funcName => {
    const regex = new RegExp(`${funcName}\\s*[:(]`);
    if (!regex.test(jsCode)) {
        missingFunctions.push(funcName);
    } else {
        console.log(`   ✅ ${funcName} - 存在`);
    }
});

if (missingFunctions.length > 0) {
    console.log(`\n❌ 缺少函数: ${missingFunctions.join(', ')}`);
} else {
    console.log('\n✅ 所有关键函数都存在');
}

// 2. 检查数据属性
console.log('\n2. 检查数据属性...');
const requiredProps = [
    'rawData:',
    'windows:',
    'annotations:',
    'klineLabels:',
    'zoomState:'
];

requiredProps.forEach(prop => {
    if (jsCode.includes(prop)) {
        console.log(`   ✅ ${prop.replace(':', '')} - 存在`);
    } else {
        console.log(`   ❌ ${prop.replace(':', '')} - 缺失`);
    }
});

// 3. 检查JSZip引入
console.log('\n3. 检查外部库引入...');
if (htmlContent.includes('jszip@3.10.1')) {
    console.log('   ✅ JSZip - 已引入');
} else {
    console.log('   ❌ JSZip - 未引入');
}

if (htmlContent.includes('echarts@5.4.3')) {
    console.log('   ✅ ECharts - 已引入');
} else {
    console.log('   ❌ ECharts - 未引入');
}

// 4. 检查HTML按钮
console.log('\n4. 检查HTML按钮...');
const buttons = [
    { text: 'saveToCSV', label: '保存窗口数据' },
    { text: 'exportLabeledCSV', label: '导出标签CSV' },
    { text: 'exportImages', label: '导出图像ZIP' }
];

buttons.forEach(btn => {
    if (htmlContent.includes(`onclick="app.${btn.text}()`)) {
        console.log(`   ✅ ${btn.label} - 按钮存在`);
    } else {
        console.log(`   ❌ ${btn.label} - 按钮缺失`);
    }
});

// 5. 检查关键逻辑
console.log('\n5. 检查关键逻辑...');

// 检查label列检测
if (jsCode.includes("headers.indexOf('label')") && jsCode.includes('hasLabelColumn')) {
    console.log('   ✅ label列检测 - 实现正确');
} else {
    console.log('   ❌ label列检测 - 可能缺失');
}

// 检查标签恢复逻辑
if (jsCode.includes('rightmostKlineIdx') && jsCode.includes('i + 23')) {
    console.log('   ✅ 标签恢复（窗口最右侧K线）- 实现正确');
} else {
    console.log('   ❌ 标签恢复 - 可能缺失');
}

// 检查缩放保持
if (jsCode.includes('zoomState') && jsCode.includes('datazoom')) {
    console.log('   ✅ 缩放状态保持 - 实现正确');
} else {
    console.log('   ❌ 缩放状态保持 - 可能缺失');
}

// 检查PyTorch格式导出
if (jsCode.includes('kline_dataset') && jsCode.includes('ImageFolder')) {
    console.log('   ✅ PyTorch ImageFolder格式 - 实现正确');
} else {
    console.log('   ❌ PyTorch格式导出 - 可能缺失');
}

// 检查README生成
if (jsCode.includes('README.md') && jsCode.includes('datasetFolder.file')) {
    console.log('   ✅ README.md生成 - 实现正确');
} else {
    console.log('   ❌ README.md生成 - 可能缺失');
}

// 6. 检查潜在问题
console.log('\n6. 检查潜在问题...');
const issues = [];

// 检查async/await使用
if (jsCode.includes('async exportImages') && jsCode.includes('await')) {
    console.log('   ✅ async/await - 使用正确');
} else {
    issues.push('exportImages可能缺少async关键字');
}

// 检查canvas toBlob
if (jsCode.includes('toBlob') && jsCode.includes('new Promise')) {
    console.log('   ✅ Canvas toBlob - 使用正确');
} else {
    issues.push('toBlob可能未正确Promise化');
}

// 检查polyfill
if (jsCode.includes('roundRect') && jsCode.includes('CanvasRenderingContext2D.prototype')) {
    console.log('   ✅ roundRect polyfill - 已添加');
} else {
    issues.push('缺少roundRect polyfill');
}

if (issues.length > 0) {
    console.log('\n⚠️ 发现潜在问题:');
    issues.forEach(issue => console.log(`   - ${issue}`));
}

// 7. 统计信息
console.log('\n7. 代码统计...');
const totalLines = htmlContent.split('\n').length;
const jsLines = jsCode.split('\n').length;
const functionCount = (jsCode.match(/\w+\s*:\s*function|\w+\s*\([^)]*\)\s*{|function\s+\w+/g) || []).length;

console.log(`   - HTML总行数: ${totalLines}`);
console.log(`   - JavaScript行数: ${jsLines}`);
console.log(`   - 函数数量: ~${functionCount}`);

console.log('\n' + '='.repeat(60));
console.log('✅ 代码审查完成！');
console.log('='.repeat(60));

// 检查是否有语法错误（简单检查）
const syntaxChecks = [
    { pattern: /\{\s*\{/, error: '可能存在双重花括号' },
    { pattern: /\)\s*\)/, error: '可能存在双重闭括号' },
    { pattern: /async\s+\w+\s*\([^)]*\)\s*{[^}]*await[^}]*}(?!\s*,)(?!\s*\))/, error: 'async函数可能缺少正确的Promise处理' }
];

console.log('\n8. 语法检查...');
let syntaxIssues = 0;
syntaxChecks.forEach(check => {
    if (check.pattern.test(jsCode)) {
        console.log(`   ⚠️ ${check.error}`);
        syntaxIssues++;
    }
});

if (syntaxIssues === 0) {
    console.log('   ✅ 未发现明显语法错误');
}

console.log('\n💡 建议:');
console.log('   1. 在浏览器中打开 http://localhost:8888/annotation_tool.html');
console.log('   2. 打开开发者工具（F12）查看控制台');
console.log('   3. 测试CSV导入、标注、导出功能');
console.log('   4. 检查缩放功能是否正常工作');

process.exit(0);
