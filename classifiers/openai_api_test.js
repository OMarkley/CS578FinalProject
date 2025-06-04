import fs from 'fs/promises';
import OpenAI from 'openai';

import dotenv from 'dotenv';
dotenv.config();

// Initialize the client (will pull API key from process.env.OPENAI_API_KEY)
const openai = new OpenAI();

//new addition
const OUTPUT_FILE = 'results2.txt';

async function classifyEmail(emailText) {
  const prompt = [
    { role: 'system', content: 'You are an email security analyst.' },
    { role: 'user', content:
      `Determine whether the following email is ‘Phishing’ or ‘Legitimate’ and reply with **exactly** one word\n\n` +
      `---\n${emailText}\n---\n` 
    }
  ];

  const resp = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: prompt,
    temperature: 0.0,
    top_p: 0
  });

  return resp.choices[0].message.content.trim();
}

async function main() {

 try {
    // 1) Clear (or create) the output file at start:
    await fs.writeFile(OUTPUT_FILE, '', 'utf-8');

    // 2) Read & split your emails:
    const data = await fs.readFile('emailss.txt', 'utf-8');
    const emails = data.split('.eml,'); // adjust delimiter if needed

    // 3) Loop, classify, log to console AND append to file
    for (let i = 0; i < emails.length; i++) {
      const email = emails[i].trim();
      if (!email) continue;

      const result = await classifyEmail(email);
      //const header = `Email ${i+1} → `;
      //console.log(/*header + */result + '\n');

      // append to results file
      await fs.appendFile(
        OUTPUT_FILE,
        /*header + */result + '\n',
        'utf-8'
      );
    }

    console.log(`All done! Results written to ${OUTPUT_FILE}`);
  } catch (err) {
    console.error('Error:', err);
  }
}

main();

