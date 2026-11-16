-- Run this script once in Supabase -> SQL Editor

create table if not exists public.setups (
  id bigint generated always as identity primary key,
  name text not null,
  price_min double precision default 0,
  price_max double precision default 99999,
  rsi_min double precision default 0,
  rsi_max double precision default 100,
  volume_multiplier double precision default 0,
  atr_pct_min double precision default 0,
  atr_pct_max double precision default 0,
  move_pct_min double precision default 0,
  move_pct_max double precision default 0,
  rule_config jsonb default '{}'::jsonb,
  visual_rules text,
  reference_image_b64 text,
  owner text default 'admin',
  created_at timestamptz default now()
);

-- Ensure columns exist if migrating from an older table version
alter table public.setups add column if not exists rule_config jsonb default '{}'::jsonb;
alter table public.setups add column if not exists owner text default 'admin';

-- Enable Row Level Security (RLS)
alter table public.setups enable row level security;

-- Allow public / anon read and write access (or adjust for Supabase auth if desired)
create policy "Allow read access to setups" on public.setups for select using (true);
create policy "Allow insert access to setups" on public.setups for insert with check (true);
create policy "Allow update access to setups" on public.setups for update using (true);
create policy "Allow delete access to setups" on public.setups for delete using (true);
