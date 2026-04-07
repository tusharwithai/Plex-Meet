require('dotenv').config();
const { Client } = require('pg');

async function main() {
  const client = new Client({
    connectionString: process.env.DATABASE_URL,
    ssl: { rejectUnauthorized: false }
  });

  try {
    console.log('Connecting to Supabase...');
    await client.connect();
    console.log('Connected successfully!');
    
    const result = await client.query(`SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'`);
    console.log('Existing tables:', result.rows.map(r => r.table_name));
  } catch (e) {
    console.error('Connection error:', e.message);
  } finally {
    await client.end();
  }
}

main();
