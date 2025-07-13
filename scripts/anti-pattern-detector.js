#!/usr/bin/env node
/**
 * Anti-Pattern Detector for Library Usage Validation
 * Scans codebase for custom implementations that should use libraries
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Define anti-patterns with detailed descriptions
const ANTI_PATTERNS = [
  // Swarm/Coordination Anti-patterns
  {
    category: 'Coordination',
    patterns: [
      { regex: /class\s+SwarmManager/g, message: 'Custom SwarmManager class - use claude-flow@alpha instead' },
      { regex: /class\s+AgentCoordinator/g, message: 'Custom AgentCoordinator - use claude-flow@alpha instead' },
      { regex: /class\s+Coordinator/g, message: 'Custom Coordinator class - use claude-flow@alpha instead' },
      { regex: /EventEmitter/g, message: 'Using EventEmitter - use claude-flow hooks instead' },
      { regex: /Observer\s+Pattern/g, message: 'Custom Observer pattern - use claude-flow hooks instead' },
      { regex: /PubSub/g, message: 'Custom PubSub - use claude-flow coordination instead' }
    ]
  },
  
  // Storage Anti-patterns
  {
    category: 'Storage',
    patterns: [
      { regex: /localStorage\./g, message: 'Using localStorage - use claude-flow memory instead' },
      { regex: /sessionStorage\./g, message: 'Using sessionStorage - use claude-flow memory instead' },
      { regex: /fs\.writeFile/g, message: 'Writing files for persistence - use claude-flow memory instead' },
      { regex: /fs\.writeFileSync/g, message: 'Writing files sync - use claude-flow memory instead' },
      { regex: /IndexedDB/g, message: 'Using IndexedDB - use claude-flow memory instead' },
      { regex: /new\s+Map\(\)/g, message: 'Using in-memory Map for state - use claude-flow persistent memory' },
      { regex: /global\.\w+\s*=/g, message: 'Using global variables - use claude-flow memory instead' }
    ]
  },
  
  // Neural Network Anti-patterns
  {
    category: 'Neural Networks',
    patterns: [
      { regex: /class\s+NeuralNetwork/g, message: 'Custom NeuralNetwork class - use ruv-FANN instead' },
      { regex: /class\s+Neuron/g, message: 'Custom Neuron class - use ruv-FANN instead' },
      { regex: /class\s+Layer/g, message: 'Custom Layer class - use ruv-FANN instead' },
      { regex: /backpropagation/gi, message: 'Custom backpropagation - use FANN.train instead' },
      { regex: /gradient\s*descent/gi, message: 'Custom gradient descent - use FANN optimization' },
      { regex: /sigmoid\s*\(/g, message: 'Custom sigmoid function - use FANN activation functions' },
      { regex: /relu\s*\(/g, message: 'Custom ReLU function - use FANN activation functions' },
      { regex: /tanh\s*\(/g, message: 'Custom tanh function - use FANN activation functions' },
      { regex: /weights\s*\*/g, message: 'Manual weight calculations - use FANN instead' }
    ]
  },
  
  // Distributed Processing Anti-patterns
  {
    category: 'Distribution',
    patterns: [
      { regex: /new\s+Worker\(/g, message: 'Using Worker threads - use DAA instead' },
      { regex: /worker_threads/g, message: 'Importing worker_threads - use DAA instead' },
      { regex: /cluster\.fork/g, message: 'Using cluster module - use DAA instead' },
      { regex: /child_process\.(fork|spawn|exec)/g, message: 'Using child_process - use DAA instead' },
      { regex: /process\.fork/g, message: 'Forking processes - use DAA instead' },
      { regex: /new\s+Thread/g, message: 'Creating threads manually - use DAA instead' }
    ]
  },
  
  // Parsing/Extraction Anti-patterns
  {
    category: 'Parsing',
    patterns: [
      { regex: /function\s+parsePDF/g, message: 'Custom PDF parser - check if library provides this' },
      { regex: /function\s+extractText/g, message: 'Custom text extractor - use library extractors' },
      { regex: /function\s+parseDocument/g, message: 'Custom document parser - use library parsers' },
      { regex: /regex.*extract/gi, message: 'Regex-based extraction - use structured library methods' },
      { regex: /\.split\(.*\)\.map\(/g, message: 'Manual parsing with split/map - check for library method' }
    ]
  },
  
  // Rule Engine Anti-patterns
  {
    category: 'Rules',
    patterns: [
      { regex: /class\s+RuleEngine/g, message: 'Custom RuleEngine - use library rule system' },
      { regex: /eval\(/g, message: 'Using eval for rules - use proper rule library' },
      { regex: /new\s+Function\(/g, message: 'Dynamic function creation - use library features' }
    ]
  }
];

// Color codes for terminal output
const colors = {
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  reset: '\x1b[0m'
};

// Statistics
let totalFiles = 0;
let filesWithViolations = 0;
let totalViolations = 0;
const violationsByCategory = {};

// Scan a single file for anti-patterns
function scanFile(filePath) {
  if (!filePath.endsWith('.js') && !filePath.endsWith('.ts')) return [];
  
  totalFiles++;
  const content = fs.readFileSync(filePath, 'utf8');
  const lines = content.split('\n');
  const violations = [];
  
  ANTI_PATTERNS.forEach(({ category, patterns }) => {
    patterns.forEach(({ regex, message }) => {
      let match;
      regex.lastIndex = 0; // Reset regex state
      
      while ((match = regex.exec(content)) !== null) {
        const lineNumber = content.substring(0, match.index).split('\n').length;
        const violation = {
          file: filePath,
          line: lineNumber,
          column: match.index - content.lastIndexOf('\n', match.index - 1),
          category,
          message,
          code: lines[lineNumber - 1].trim(),
          match: match[0]
        };
        violations.push(violation);
        totalViolations++;
        
        // Track by category
        violationsByCategory[category] = (violationsByCategory[category] || 0) + 1;
      }
    });
  });
  
  if (violations.length > 0) {
    filesWithViolations++;
  }
  
  return violations;
}

// Recursively scan directory
function scanDirectory(dir, baseDir = dir) {
  const allViolations = [];
  
  function walkDir(currentDir) {
    const files = fs.readdirSync(currentDir);
    
    files.forEach(file => {
      const filePath = path.join(currentDir, file);
      const stat = fs.statSync(filePath);
      
      // Skip certain directories
      if (stat.isDirectory()) {
        if (!file.startsWith('.') && 
            file !== 'node_modules' && 
            file !== 'dist' && 
            file !== 'build' &&
            file !== 'coverage') {
          walkDir(filePath);
        }
      } else if (stat.isFile()) {
        const violations = scanFile(filePath);
        if (violations.length > 0) {
          allViolations.push({
            file: path.relative(baseDir, filePath),
            violations
          });
        }
      }
    });
  }
  
  walkDir(dir);
  return allViolations;
}

// Format violation for display
function formatViolation(violation) {
  return `  ${colors.yellow}${violation.file}:${violation.line}:${violation.column}${colors.reset}
  ${colors.red}✗${colors.reset} ${violation.message}
  ${colors.blue}>${colors.reset} ${violation.code}
  ${colors.blue}  Found:${colors.reset} "${violation.match}"`;
}

// Main execution
function main() {
  console.log(`${colors.blue}🔍 Anti-Pattern Detector for Library Usage${colors.reset}`);
  console.log('=' .repeat(50));
  
  const srcDir = path.join(process.cwd(), 'src');
  
  if (!fs.existsSync(srcDir)) {
    console.error(`${colors.red}Error: src/ directory not found${colors.reset}`);
    process.exit(1);
  }
  
  console.log(`Scanning ${srcDir}...\n`);
  const results = scanDirectory(srcDir, process.cwd());
  
  // Display results
  if (results.length === 0) {
    console.log(`${colors.green}✅ No anti-patterns detected!${colors.reset}`);
    console.log('All code follows library usage guidelines.\n');
  } else {
    console.log(`${colors.red}❌ Anti-patterns detected:${colors.reset}\n`);
    
    // Group by category
    const byCategory = {};
    results.forEach(({ file, violations }) => {
      violations.forEach(v => {
        if (!byCategory[v.category]) byCategory[v.category] = [];
        byCategory[v.category].push({ ...v, file });
      });
    });
    
    // Display by category
    Object.entries(byCategory).forEach(([category, violations]) => {
      console.log(`${colors.yellow}${category} Violations (${violations.length}):${colors.reset}`);
      violations.forEach(v => {
        console.log(formatViolation(v));
        console.log('');
      });
    });
  }
  
  // Summary statistics
  console.log('=' .repeat(50));
  console.log(`${colors.blue}Summary:${colors.reset}`);
  console.log(`  Total files scanned: ${totalFiles}`);
  console.log(`  Files with violations: ${filesWithViolations}`);
  console.log(`  Total violations: ${totalViolations}`);
  
  if (totalViolations > 0) {
    console.log(`\n${colors.yellow}Violations by category:${colors.reset}`);
    Object.entries(violationsByCategory).forEach(([cat, count]) => {
      console.log(`  ${cat}: ${count}`);
    });
    
    console.log(`\n${colors.red}⚠️  Please refactor to use the required libraries:${colors.reset}`);
    console.log('  • claude-flow@alpha - for coordination and memory');
    console.log('  • ruv-FANN - for neural networks');
    console.log('  • DAA - for distributed processing');
    
    // Store results in memory
    try {
      execSync(`npx claude-flow@alpha memory store "validation-violations" "${JSON.stringify({
        totalViolations,
        violationsByCategory,
        timestamp: new Date().toISOString()
      })}" --namespace validation`);
      console.log(`\n${colors.blue}Validation results stored in claude-flow memory${colors.reset}`);
    } catch (e) {
      // Ignore if claude-flow is not available
    }
    
    process.exit(1);
  }
  
  process.exit(0);
}

// Run if executed directly
if (require.main === module) {
  main();
}

module.exports = { scanFile, scanDirectory, ANTI_PATTERNS };